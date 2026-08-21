# SiDLIF: A Scale-Invariant Deep Learning-Based Inversion Framework for Surface-Wave Dispersion Curves via Physics-Guided Normalization

**SiDLIF** is a physics-guided deep learning framework for inverting **fundamental-mode Rayleigh-wave dispersion curves** into shear-wave velocity ($V_S$) profiles.

By exploiting the scaling properties of surface-wave dispersion, SiDLIF transforms dispersion curves and their corresponding $V_S$ profiles into **dimensionless representations**. This decouples the learned inverse mapping from absolute depth and velocity scales, allowing the same pretrained model to be applied across substantially different physical scales without site-specific retraining.

The framework also incorporates the **Point-wise Additive Dispersion Inversion Transformer (PADIT)** to accommodate dispersion curves with variable frequency coverage, together with a misfit-constrained grid-search workflow that identifies ensembles of physically admissible solutions through forward-model validation.

This repository provides the implementation associated with the manuscript:

**“SiDLIF: A Scale-Invariant Deep Learning-Based Inversion Framework for Surface-Wave Dispersion Curves via Physics-Guided Normalization”**

> **Note:** The repository URL retains the original U-SWIFT name for continuity. The current framework and manuscript are named **SiDLIF**.

---

## Key Features

* **Scale-invariant inversion**
  Physics-guided dimensionless normalization removes the dependence of the learned inverse mapping on absolute depth and velocity scales.

* **Variable-length dispersion curves**
  PADIT accommodates dispersion curves with different frequency coverages and different numbers of sampled points.

* **Two structural priors**
  Two pretrained PADIT models are provided:

  * **Monotonic PADIT**, trained using monotonically increasing $V_S$ profiles.
  * **LVZ-inclusive PADIT**, trained using both monotonic profiles and profiles containing low-velocity zones (LVZs).

* **Forward-model-constrained solutions**
  Every candidate $V_S$ profile is evaluated by forward modeling its theoretical dispersion curve.

* **Accepted-model ensembles**
  Candidate profiles satisfying a prescribed dispersion-curve misfit threshold are retained as an ensemble of physically admissible solutions.

* **Minimal manual parameterization**
  Users do not need to prescribe a site-specific number of layers. The main search parameters are the half-space depth $z_{hs}$ and half-space shear-wave velocity $V_{S,hs}$.

---

## Method Overview

### 1. Dimensionless Normalization

For a subsurface model with half-space depth $z_{hs}$ and half-space shear-wave velocity $V_{S,hs}$, depth and shear-wave velocity are normalized as

```math
z^* = \frac{z}{z_{hs}},
```

```math
V_S^* = \frac{V_S}{V_{S,hs}}.
```

The corresponding dispersion curve is normalized according to

```math
c^* = \frac{c}{V_{S,hs}},
```

```math
f^* = \frac{f z_{hs}}{V_{S,hs}},
```

where $f$ is frequency and $c$ is Rayleigh-wave phase velocity.

The resulting normalized profile has

```math
z_{hs}^* = 1,
\qquad
V_{S,hs}^* = 1.
```

This dimensionless representation allows PADIT to learn the nonlinear relationship between dispersion curves and velocity structures independently of their absolute physical scales.

---

### 2. Grid Search Over Half-Space Parameters

In field applications, $z_{hs}$ and $V_{S,hs}$ are generally unknown a priori. SiDLIF therefore evaluates multiple candidate pairs over prescribed search ranges.

The maximum resolved wavelength is defined as

```math
\lambda_{\max}
=
\max\left(\frac{c}{f}\right).
```

In the absence of additional prior information, practical initial search intervals are

```math
z_{hs}
\in
[0.2\lambda_{\max},\,0.7\lambda_{\max}],
```

and

```math
V_{S,hs}
\in
[1.05c_{\max},\,3c_{\max}],
```

where $c_{\max}$ denotes the phase velocity associated with the longest resolved wavelength.

These ranges are intended as initial bounds and can be modified when additional geological or geophysical information is available.

In the example workflow, $z_{hs}$ and $V_{S,hs}$ are each sampled at 200 points, yielding 40,000 candidate parameter pairs.

---

### 3. PADIT Inversion

Each normalized dispersion curve is passed to the **Point-wise Additive Dispersion Inversion Transformer (PADIT)**.

PADIT contains:

* a point-wise feature extractor;
* positional encoding;
* six pre-LayerNorm Transformer encoder layers;
* eight attention heads in each Transformer layer;
* a feed-forward hidden dimension of 1024;
* sum aggregation over the sequence dimension;
* a regression head producing 100 normalized $V_S^*$ values;
* a Sigmoid output activation.

The regression head predicts the $V_S^*$ values of 100 equally spaced layers above the half-space. Each layer has normalized thickness

```math
t^* = 0.01.
```

The normalized half-space shear-wave velocity is fixed at

```math
V_{S,hs}^* = 1.
```

The Sigmoid output constrains the predicted normalized velocities to the interval between 0 and 1.

---

### 4. Denormalization

For every candidate pair $(z_{hs},V_{S,hs})$, the predicted dimensionless profile is transformed back to physical units using

```math
z = z^* z_{hs},
```

and

```math
V_S = V_S^* V_{S,hs}.
```

Each candidate pair therefore produces one candidate $V_S$ profile at the physical depth and velocity scale of the target structure.

---

### 5. Forward-Model Validation

For every candidate $V_S$ profile, a theoretical fundamental-mode Rayleigh-wave dispersion curve is calculated by forward modeling.

The agreement between the theoretical and observed dispersion curves is quantified using

```math
\mathrm{misfit}
=
\sqrt{
\frac{1}{m}
\sum_{j=1}^{m}
\left(
\frac{
c_j^{\mathrm{obs}}-c_j^{\mathrm{theo}}
}{
\sigma_j
}
\right)^2
}.
```

Here:

* $c_j^{\mathrm{obs}}$ is the observed phase velocity;
* $c_j^{\mathrm{theo}}$ is the theoretical phase velocity;
* $\sigma_j$ is the measurement uncertainty;
* $m$ is the number of dispersion-curve points.

Candidate profiles satisfying

```math
\mathrm{misfit} < 1
```

are retained as **accepted models**.

The candidate with the lowest misfit is identified as the **best-fit model**, while the full accepted-model ensemble represents the range of velocity structures consistent with the observed dispersion curve under the adopted search bounds, parameterization, and structural prior.

---

## Pretrained PADIT Models

Two pretrained PADIT models are provided with SiDLIF.

| Model               | Weight file               | Structural prior                                 |
| ------------------- | ------------------------- | ------------------------------------------------ |
| Monotonic PADIT     | `PADIT_monotonic.pth`     | Monotonically increasing $V_S$ profiles          |
| LVZ-inclusive PADIT | `PADIT_LVZ_inclusive.pth` | Allows low-velocity zones and velocity reversals |

### Monotonic PADIT

The monotonic model is appropriate when an overall increase in $V_S$ with depth is considered a reasonable structural constraint.

Because it imposes a stronger structural prior, it generally produces a more restricted admissible solution space.

### LVZ-Inclusive PADIT

The LVZ-inclusive model should be considered when low-velocity zones or velocity reversals may occur.

Because it represents a broader range of possible velocity structures, its accepted-model ensemble may also be broader.

The two models therefore represent **different structural priors** rather than competing models for which one is universally preferable.

---

## Downloading the Pretrained Models

The pretrained `.pth` files are distributed through the **GitHub Releases** page rather than stored directly in the source-code repository.

Download:

```text
PADIT_monotonic.pth
PADIT_LVZ_inclusive.pth
```

and place them in the `models/` directory:

```text
models/
├── PADIT_monotonic.pth
└── PADIT_LVZ_inclusive.pth
```

---

## Quick Start

### 1. Download the Repository

Download the repository using **Code → Download ZIP**, or clone it using Git:

```bash
git clone https://github.com/CTJ-UT/U-SWIFT-A-Unified-Surface-Wave-Inversion-Framework-with-Transformer.git
cd U-SWIFT-A-Unified-Surface-Wave-Inversion-Framework-with-Transformer
```

### 2. Download the Pretrained Models

Download the two PADIT weight files from the latest GitHub Release and place them inside

```text
models/
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Prepare the Dispersion Curve

An example dispersion curve is provided in the `examples/` directory.

The input data contain three columns:

```text
frequency (Hz)    phase velocity (m/s)    standard deviation (m/s)
```

The data are loaded in `main.ipynb` using

```python
data = np.loadtxt("examples/example_01.txt")

f = data[:, 0]
vr = data[:, 1]
vr_std = data[:, 2]
```

where:

* `f` is frequency;
* `vr` is Rayleigh-wave phase velocity;
* `vr_std` is the corresponding standard deviation.

If measurement standard deviations are not directly available, users should define `vr_std` according to an appropriate uncertainty estimate for their dispersion measurements.

---

### 5. Select the PADIT Model

The example notebook uses the monotonic PADIT model by default:

```python
model_path = "models/PADIT_monotonic.pth"  # Replace with "models/PADIT_LVZ_inclusive.pth" to use the LVZ-inclusive model.
```

---

### 6. Run the Inversion

Open

```text
main.ipynb
```

and execute the cells sequentially.

The notebook performs the complete SiDLIF workflow:

1. load and sort the observed dispersion curve;
2. define the search ranges for $z_{hs}$ and $V_{S,hs}$;
3. construct candidate half-space parameter pairs;
4. normalize the dispersion curves;
5. perform PADIT inference;
6. denormalize the predicted $V_S$ profiles;
7. calculate theoretical dispersion curves by forward modeling;
8. evaluate dispersion-curve misfits;
9. identify the best-fit model and accepted-model ensemble;
10. visualize the inversion results.

---

## Adjusting the Search Space

The default half-space velocity range in `main.ipynb` is defined as

```python
vs_bounds_halfspace = np.array([
    1.05 * np.max(vr),
    3.0 * np.max(vr)
])
```

The default half-space depth range is

```python
depth_bounds_halfspace = np.array([
    0.2,
    0.7
]) * np.max(vr / f)
```

The numbers of sampled half-space depths and velocities can also be modified:

```python
depth, vs_hs, f_vr_resampled = scale_and_resample_dc(
    f,
    vr,
    vs_bounds_halfspace,
    depth_bounds_halfspace,
    nd=200,
    nv=200,
    v_scale="log",
    d_scale="log"
)
```

Here:

* `nd` is the number of sampled half-space depths;
* `nv` is the number of sampled half-space shear-wave velocities.

Increasing `nd` and `nv` provides denser coverage of the search space but also increases computational cost.

If no accepted models are obtained, the search ranges should be reconsidered or broadened.

If independent geological or geophysical constraints are available, they can be used to define narrower and more appropriate search ranges.

---

## Output

The inversion workflow produces:

* candidate $V_S$ profiles for the sampled $(z_{hs},V_{S,hs})$ pairs;
* corresponding forward-modeled dispersion curves;
* dispersion-curve misfit values;
* the best-fit $V_S$ profile;
* the accepted-model ensemble satisfying the prescribed misfit threshold;
* the corresponding accepted theoretical dispersion curves.

The accepted-model ensemble provides a practical way to examine inversion nonuniqueness and the influence of the structural prior represented by the selected PADIT model.

---

## Repository Structure

```text
.
├── assets/
├── examples/
│   └── example_01.txt
├── models/
│   └── .gitkeep
├── src/
│   ├── aggregation_model.py
│   └── utils.py
├── .gitignore
├── LICENSE
├── README.md
├── main.ipynb
└── requirements.txt
```

### `main.ipynb`

Example notebook implementing the complete SiDLIF inversion workflow.

### `src/aggregation_model.py`

Implementation of the PADIT neural-network architecture.

### `src/utils.py`

Utility functions used for:

* dimensionless normalization;
* dispersion-curve resampling;
* PADIT inference;
* profile denormalization;
* Rayleigh-wave forward modeling;
* misfit calculation;
* accepted-model identification and result processing.

### `examples/example_01.txt`

Example fundamental-mode Rayleigh-wave dispersion curve.

### `models/`

Local directory for pretrained PADIT weights downloaded from GitHub Releases.

The large `.pth` files are intentionally not stored directly in the source-code repository.

---

## Current Scope and Assumptions

The current pretrained PADIT models were developed and validated for:

* fundamental-mode Rayleigh-wave dispersion curves;
* one-dimensional horizontally layered subsurface structures;
* 100 normalized layers above the half-space;
* normalized layer thickness $t^*=0.01$;
* a fixed Poisson's ratio of $1/3$ in the synthetic training models;
* constant density in the synthetic training models.

The present implementation focuses on inversion for $V_S$.

The 100-layer numerical discretization should **not** be interpreted as the intrinsic geophysical spatial resolution of the inversion. Actual resolution is primarily controlled by the information content and frequency coverage of the observed dispersion curve.

The current pretrained PADIT models were trained and validated using fundamental-mode Rayleigh-wave dispersion curves alone. Future extensions may incorporate additional elastic parameters, higher-mode dispersion information, or other geophysical observations.

---

## Reproducibility

The source code, example data, and pretrained model weights are provided to facilitate reproduction and application of the SiDLIF workflow.

For reproducible use:

1. use the source code associated with the corresponding GitHub Release;
2. use the pretrained PADIT weights distributed with that release;
3. keep the model file names unchanged;
4. record the $z_{hs}$ and $V_{S,hs}$ search ranges used for each inversion;
5. record the selected PADIT structural prior and misfit threshold.

A versioned research archive is also maintained through Zenodo.

---

## Citation

If you use SiDLIF in your research, please cite the associated manuscript:

**Tianjian Cheng, Hongrui Xu, Jiayu Feng, Qiaomu Qi, Xiongyu Hu, and Chaofan Yao.**

**“SiDLIF: A Scale-Invariant Deep Learning-Based Inversion Framework for Surface-Wave Dispersion Curves via Physics-Guided Normalization.”**

Complete journal, volume, page, and DOI information will be added after publication.

---

## License

This project is distributed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

See the `LICENSE` file for details.

---

## Contact

For questions regarding SiDLIF or its implementation, please contact:

**Tianjian Cheng**
Faculty of Geosciences and Engineering
Southwest Jiaotong University
Email: `tjcheng.ok@163.com`

**Hongrui Xu**
Faculty of Geosciences and Engineering
Southwest Jiaotong University
Email: `hongrui_xu@swjtu.edu.cn`
