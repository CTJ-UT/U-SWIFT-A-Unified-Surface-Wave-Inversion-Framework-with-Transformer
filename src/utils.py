import numpy as np
import torch
from collections import defaultdict
from disba import PhaseDispersion
from aggregation_model import Config, LayerNormSumAggregationModel
from tqdm import tqdm
from joblib import Parallel, delayed

FREQ_LOG_MIN = -2.3
FREQ_LOG_MAX = 2.0

def scale_and_resample_dc(f, vr, vs_bounds, depth_bounds, nv = 400, nd = 400, v_scale = 'linear', d_scale = 'linear'):
    """
    Scales and resamples dispersion curve data over specified velocity and depth ranges.
    This function rescales the input frequency and velocity data according to the specified
    velocity and depth bounds, using either linear or logarithmic scaling. It then resamples
    the data onto a new frequency grid, interpolates the velocity values, and returns the
    scaled parameters and resampled data.
    Parameters
    ----------
    f : array-like
        1D array of input frequencies.
    vr : array-like
        1D array of input phase velocities corresponding to `f`.
    vs_bounds : tuple of float
        (min, max) bounds for shear wave velocity (Vs) scaling.
    depth_bounds : tuple of float
        (min, max) bounds for depth scaling.
    nv : int, optional
        Number of velocity samples to generate (default is 400).
    nd : int, optional
        Number of depth samples to generate (default is 400).
    v_scale : {'linear', 'log'}, optional
        Scaling type for velocity axis (default is 'linear').
    d_scale : {'linear', 'log'}, optional
        Scaling type for depth axis (default is 'linear').
    Returns
    -------
    depth : ndarray
        1D array of depth for the selected (valid) grid points.
    vs_hs : ndarray
        1D array of velocity for the selected (valid) grid points.
    f_vr_resampled : list of ndarray
        List where each element is a 2D array of shape (N, 2), containing resampled
        frequency and velocity pairs for each valid grid point.
    Notes
    -----
    - The function filters out grid points where the scaled frequency range does not
      overlap with the target frequency range [10^-2.3, 10^2] Hz.
    - Interpolation is performed in log-frequency space.
    - Output `f_vr_resampled` contains only valid (non-NaN) frequency-velocity pairs
      for each grid point.
    """
    
    if v_scale == 'linear':
        vs_hs = np.linspace(vs_bounds[0], vs_bounds[1], nv)
    elif v_scale == 'log':
        vs_hs = np.logspace(np.log10(vs_bounds[0]), np.log10(vs_bounds[1]), nv)
    else:
        raise ValueError("v_scale must be 'linear' or 'log'")
    
    if d_scale == 'linear':
        depths = np.linspace(depth_bounds[0], depth_bounds[1], nd)
    elif d_scale == 'log':
        depths = np.logspace(np.log10(depth_bounds[0]), np.log10(depth_bounds[1]), nd)
    else:
        raise ValueError("d_scale must be 'linear' or 'log'")

    f = np.asarray(f).ravel()  
    vr = np.asarray(vr).ravel()    
    depth = np.tile(depths, len(vs_hs))
    vs_hs = np.repeat(vs_hs, len(depths))

    scale = depth / vs_hs 
    f_lo, f_hi = f.min(), f.max()           
    keep = np.where((scale * f_hi >= 10**-2.3) & (scale * f_lo <= 10**2))[0]
    depth, vs_hs = depth[keep], vs_hs[keep]
    scale = depth / vs_hs

    f_all = np.logspace(-2.3, 2, 256)
    log_f_all = np.log(f_all)
    log_scale = np.log(scale)
    log_f_reverse_scaled = log_f_all[None, :] - log_scale[:, None]
    vr_reverse_scaled = np.interp(
        log_f_reverse_scaled.ravel(),
        np.log(f),
        vr,
        left=np.nan,
        right=np.nan,
    ).reshape(log_f_reverse_scaled.shape)
    vr_resampled = vr_reverse_scaled / vs_hs[:, None]

    valid_mask = ~np.isnan(vr_resampled)
    _, cols = np.where(valid_mask)
    stacked_data = np.column_stack([f_all[cols], vr_resampled[valid_mask]])
    split_indices = valid_mask.sum(axis=1).cumsum()[:-1]
    f_vr_resampled = np.split(stacked_data, split_indices, axis=0)
    
    return depth, vs_hs, f_vr_resampled

def denormalize_dc(vs_predict, depth, vs_hs):
    """
    Denormalizes predicted Vs values and computes inverted thickness.
    The model predicts normalized values: vs_norm = vs / vs_hs and
    thickness_norm = thickness / depth. This function recovers the
    physical values by multiplying back the normalization factors.
    Parameters
    ----------
    vs_predict : np.ndarray
        Predicted normalized Vs values (vs / vs_hs), shape (n, 101).
    depth : np.ndarray
        Depth values used for normalization, shape (n,).
    vs_hs : np.ndarray
        Half-space Vs values used for normalization, shape (n,).
    Returns
    -------
    thickness_inverted : np.ndarray
        Inverted thickness values in physical units, shape (n, 101).
    vs_inverted : np.ndarray
        Inverted Vs values in physical units, shape (n, 101).
    """
    
    n_layers = vs_predict.shape[1]
    normalized_thickness = np.concatenate([np.full(n_layers - 1, 0.01), [0.0]])
    thickness_inverted = normalized_thickness * depth.reshape(-1, 1)
    vs_inverted = vs_predict * vs_hs.reshape(-1, 1)

    return thickness_inverted, vs_inverted

def run_prediction(model_path, input_data):
    """
    Runs prediction on input data using a pre-trained model.
    This function loads a trained model, preprocesses the input data, groups samples
    by sequence length, and performs batch prediction.
    Device selection (CPU/GPU) is handled automatically, and progress is displayed with a progress bar.

    Parameters
    ----------
    model_path : str
        Path to the saved model checkpoint file.
    input_data : list of np.ndarray
        List of input samples, each as a 2D numpy array of shape (sequence_length, 2),
        where columns represent frequency and velocity.

    Returns
    -------
    np.ndarray
        Array of model predictions corresponding to each input sample.

    Notes
    -----
    - Preprocesses input data with the same normalization used during training:
      frequency = (log10(f) - FREQ_LOG_MIN) / (FREQ_LOG_MAX - FREQ_LOG_MIN),
      velocity is used as-is because it is already vr / vs_hs.
    - Handles device selection (CPU/GPU) automatically.
    - Displays progress using a progress bar.
    """
    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model ---
    print(f"Input data: {len(input_data)} samples")

    print("Loading model...")
    cfg = Config()
    model = LayerNormSumAggregationModel(cfg, use_transformer=True).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(key.startswith('module.') for key in state_dict):
        state_dict = {key.replace('module.', '', 1): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Data Preprocessing and Grouping ---
    print("Preprocessing data...")
    length_groups = defaultdict(list)
    for i, sample in enumerate(input_data):
        length = len(sample)
        frequencies = (np.log10(sample[:, 0]) - FREQ_LOG_MIN) / (FREQ_LOG_MAX - FREQ_LOG_MIN)
        velocities = sample[:, 1]
        points = np.column_stack([frequencies, velocities])
        length_groups[length].append((i, points))

    # --- Create Batches and Predict Directly ---
    print("Starting prediction...")
    results = [None] * len(input_data)
    
    with torch.no_grad():
        progress_bar = tqdm(total=len(input_data), desc="Prediction Progress")
        for length, items in length_groups.items():
            indices, points_list = zip(*items)
            points_array = np.array(points_list)

            for i in range(0, len(points_array), cfg.batch_size):
                end_idx = min(i + cfg.batch_size, len(points_array))

                batch_points = torch.FloatTensor(points_array[i:end_idx]).to(device)
                batch_indices = indices[i:end_idx]
                
                batch_predictions = model(batch_points)
                
                for j, orig_idx in enumerate(batch_indices):
                    results[orig_idx] = batch_predictions[j].cpu().numpy()

                progress_bar.update(len(batch_indices))
        progress_bar.close()

    # --- Finalize Results ---
    vs_predict = np.array(results)
    
    print("Prediction complete!")
    print(f"Final result shape: {vs_predict.shape}")  
    return vs_predict

def compute_0mode_R_dispersion(thickness, vs, f):
    """
    Compute the fundamental mode (0-mode) Rayleigh wave phase velocities for a given layered model.
    This function attempts to compute the Rayleigh wave phase velocity dispersion curve for the 
    fundamental mode (mode 0) using a range of resolution parameters. If all attempts fail, 
    it returns an array of zeros and prints a warning.
    Note: thickness and velocities are scaled by the same factor before calling
    disba for numerical stability; the output phase velocities are scaled back.
    Parameters
    ----------
    thickness : array_like
        Layer thicknesses (in the same units as velocity, e.g., meters).
    vs : array_like
        Shear wave velocities for each layer (in the same units as thickness per second, e.g., m/s).
    f : array_like
        Frequencies at which to compute the dispersion curve (Hz).
    Returns
    -------
    velocities : ndarray
        Computed Rayleigh wave phase velocities at the input frequencies. If computation fails, 
        returns an array of zeros with the same length as `f`.
    """

    periods = 1.0 / f[::-1]
    density = np.ones(len(thickness))*2
    velocity_model = np.vstack((thickness, 2*vs, vs, density))

    # Define the list of resolutions to try
    resolutions = [0.05, 0.005, 0.0005]
    for res in resolutions:
        try:
            pd = PhaseDispersion(*velocity_model, dc=res)
            dc = pd(periods, mode=0, wave="rayleigh")
            velocities = dc.velocity[::-1]
            return velocities
        except Exception as e:
            continue
    
    print(f"Warning: All resolutions failed for the model. Returning zeros.")
    return np.zeros(len(f))

def forward_parallel(vs_profiles, f_scaled, n_jobs=-1):
    """
    Computes forward modeling of phase velocities for multiple Vs profiles in parallel.
    Parameters
    ----------
    vs_profiles : np.ndarray
        Array of shape (n_models, n_layers) containing shear wave velocity profiles for each model.
    f_scaled : np.ndarray
        Array of shape (n_models, n_frequencies) containing scaled frequencies for each model.
    n_jobs : int, optional
        Number of parallel jobs to run. Default is -1 (use all available cores).
    Returns
    -------
    np.ndarray
        Array of shape (n_models, n_frequencies) containing computed phase velocities for each model.
    """
    
    periods = 1.0 / f_scaled[:,::-1]
    n_models = vs_profiles.shape[0]
    scale = 100
    vs_profiles_scaled = vs_profiles * scale
    density = np.ones(vs_profiles.shape[1])
    thickness = np.concatenate([np.full(vs_profiles.shape[1] - 1, 0.01), [0.0]]) * scale
    resolutions = [0.05, 0.005, 0.0005, 0.00005, 0.00001]
    vp_profiles_scaled = 2 * vs_profiles_scaled

    def _forward(i):
        for res in resolutions:
            try:
                pd = PhaseDispersion(thickness, vp_profiles_scaled[i], vs_profiles_scaled[i], density, dc=res)
                dc = pd(periods[i], mode=0, wave="rayleigh")
                return dc.velocity[::-1]
            except Exception as e:
                continue
    
        print("Warning: All resolutions failed for the model. Returning zeros.")
        return np.zeros(f_scaled.shape[1])

    vrs = Parallel(n_jobs=n_jobs, verbose=1, prefer="threads")(
        delayed(_forward)(i) for i in range(n_models)
    )

    return np.array(vrs) / scale

def find_qualified_indices_and_rank(vr, vr_err, vr_models, limit=1):
    """
    Identify indices of model predictions that fit observed values within a specified misfit limit, and rank them by misfit.
    Parameters
    ----------
    vr : array_like
        Observed values to be matched.
    vr_err : array_like
        Uncertainties (standard deviations) associated with the observed values.
    vr_models : array_like
        Model predictions to be compared with the observed values. Should have shape (n_models, n_values).
    limit : float, optional
        Maximum allowed misfit for a model to be considered qualified. Default is 1.
    Returns
    -------
    qualified_indices : ndarray
        Indices of models whose misfit is less than or equal to the specified limit, sorted by increasing misfit.
    misfit : ndarray
        Misfit values corresponding to the qualified models, sorted in ascending order.
    Notes
    -----
    The misfit is calculated as the root mean square of the normalized residuals between 
    the model predictions and the observed values.
    """

    misfit = np.sqrt(np.mean(((vr_models - vr) / vr_err)**2, axis = 1))
    qualified_indices = np.where(misfit <= limit)[0]
    misfit = misfit[qualified_indices]
    sort_order = np.argsort(misfit)
    misfit = misfit[sort_order]
    qualified_indices = qualified_indices[sort_order]

    return qualified_indices, misfit

def get_depth_for_plot(thickness_inverted):
    """
    Calculate cumulative depth values for plotting based on inverted thickness values.

    Parameters
    ----------
    thickness_inverted : array_like
        Array of inverted thickness values. Can be 1D or 2D. Each row (if 2D) represents a set of thicknesses.

    Returns
    -------
    depth_plot : ndarray
        Array of cumulative depth values suitable for plotting. The shape matches the input:
        - If input is 1D, returns a 1D array.
        - If input is 2D, returns a 2D array with the same number of rows.

    Notes
    -----
    The function ensures that the depth values are repeated and aligned for step plotting.
    The last depth value is set to the maximum to ensure proper plot boundaries.
    """
    
    original_ndim = thickness_inverted.ndim
    thickness_inverted = np.atleast_2d(thickness_inverted)
    depth_plot = np.cumsum(thickness_inverted, axis=1)
    depth_plot = np.repeat(depth_plot, 2, axis=1)
    depth_plot = depth_plot[:, :-1]
    depth_plot[:,-1] = 1.05 * np.max(depth_plot[:,-1])
    depth_plot = np.hstack((np.zeros((depth_plot.shape[0], 1)), depth_plot))
    if original_ndim == 1:
        return depth_plot.squeeze()
    else:
        return depth_plot
