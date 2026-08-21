# SiDLIF: A Scale-Invariant Deep Learning-Based Inversion Framework for Surface-Wave Dispersion Curves via Physics-Guided Normalization

SiDLIF is a physics-guided deep learning framework for inverting **fundamental-mode Rayleigh-wave dispersion curves** into shear-wave velocity ($V_S$) profiles.

By exploiting the scaling properties of surface-wave dispersion, SiDLIF transforms dispersion curves and their corresponding $V_S$ profiles into **dimensionless representations**. This decouples the learned inverse mapping from absolute depth and velocity scales, allowing the same pretrained model to be applied to problems ranging from shallow near-surface investigations to crustal-scale imaging without site-specific retraining.

This repository provides the implementation of the manuscript:

**“SiDLIF: A Scale-Invariant Deep Learning-Based Inversion Framework for Surface-Wave Dispersion Curves via Physics-Guided Normalization”**

> **Note:** The repository URL retains the original U-SWIFT name for continuity. The current framework and manuscript are named **SiDLIF**.

---

## Key Features

* **Scale-invariant inversion**
  Physics-guided dimensionless normalization removes the dependence of the neural network on absolute depth and velocity scales.

* **Variable-length dispersion curves**
  The Point-wise Additive Dispersion Inversion Transformer (**PADIT**) accommodates dispersion curves with different frequency coverages and numbers of sampled points.

* **Two structural priors**
  Two pretrained PADIT models are provided:

  * a **monotonic model**, trained using monotonically increasing $V_S$ profiles;
  * an **LVZ-inclusive model**, trained using both monotonic profiles and profiles containing low-velocity zones (LVZs).

* **Forward-model-constrained inversion**
  Candidate $V_S$ profiles predicted by PADIT are evaluated by forward modeling their theoretical dispersion curves.

* **Accepted-model ensembles**
  Candidate models satisfying a prescribed dispersion-curve misfit threshold are retained as an ensemble of physically admissible solutions.

* **Minimal manual parameterization**
  Users do not need to prescribe a site-specific number of layers. The main search parameters are the half-space depth $z_{hs}$ and half-space shear-wave velocity $V_{S,hs}$.

---

## Method Overview

### 1. Dimensionless Normalization

For a subsurface model with half-space depth $z_{hs}$ and half-space shear-wave velocity $V_{S,hs}$, the velocity profile is normalized as

$$
z^* = \frac{z}{z_{hs}},
$$

$$
V_S^* = \frac{V_S}{V_{S,hs}}.
$$

The corresponding dispersion curve is normalized as

$$
c^* = \frac{c}{V_{S,hs}},
$$

$$
f^* = \frac{f z_{hs}}{V_{S,hs}},
$$

where $f$ is frequency and $c$ is Rayleigh-wave phase velocity.

The resulting normalized profile has

$$
z_{hs}^* = 1, \qquad V_{S,hs}^* = 1.
$$

This dimensionless representation allows PADIT to learn the nonlinear relationship between dispersion curves and velocity structures independently of their absolute physical scales.

---

### 2. Grid Search Over Half-Space Parameters

Because $z_{hs}$ and $V_{S,hs}$ are generally unknown a priori, SiDLIF evaluates multiple candidate pairs over user-defined search ranges.

In the absence of additional prior information, the initial search ranges can be estimated from the maximum resolved wavelength

$$
\lambda_{\max} = \max\left(\frac{c}{f}\right).
$$

The default ranges used in the example are

$$
z_{hs} \in [0.2\lambda_{\max},\ 0.7\lambda_{\max}],
$$

and

$$
V_{S,hs} \in [1.05c_{\max},\ 3c_{\max}],
$$

where $c_{\max}$ is the phase velocity corresponding to the maximum resolved wavelength.

These intervals are intended as practical initial ranges and can be modified when additional geological or geophysical information is available.

The default example samples 200 values for $z_{hs}$ and 200 values for $V_{S,hs}$, producing 40,000 candidate parameter pairs.

---

### 3. PADIT Inversion

Each normalized dispersion curve is passed to the **Point-wise Additive Dispersion Inversion Transformer (PADIT)**.

PADIT consists of:

* a point-wise feature extractor;
* positional encoding;
* six pre-LayerNorm Transformer encoder layers;
* sum aggregation over the sequence dimension;
* a regression head producing 100 normalized $V_S^*$ values;
* a Sigmoid activation at the output layer.

The output represents 100 equally spaced layers above the half-space, each with normalized thickness

$$
t^* = 0.01.
$$

The half-space velocity is fixed at

$$
V_{S,hs}^* = 1.
$$

---

### 4. Denormalization

The predicted dimensionless velocity profile is transformed back to the original physical scale according to

$$
z = z^* z_{hs},
$$

and

$$
V_S = V_S^* V_{S,hs}.
$$

Each candidate pair $(z_{hs}, V_{S,hs})$ therefore produces one candidate $V_S$ profile in physical units.

---

### 5. Forward-Model Validation

For each candidate $V_S$ profile, the corresponding theoretical fundamental-mode Rayleigh-wave dispersion curve is calculated by forward modeling.

The misfit is defined as

$$
\mathrm{misfit}
===============

\sqrt{
\frac{1}{m}
\sum_{j=1}^{m}
\left(
\frac{
c_j^{\mathrm{obs}} - c_j^{\mathrm{theo}}
}{
\sigma_j
}
\right)^2
},
$$

where:

* $c_j^{\mathrm{obs}}$ is the observed phase velocity;
* $c_j^{\mathrm{theo}}$ is the theoretical phase velocity;
* $\sigma_j$ is the measurement uncertainty;
* $m$ is the number of dispersion-curve points.

Candidate profiles satisfying

$$
\mathrm{misfit} < 1
$$

are retained as **accepted models**.

The model with the lowest misfit is identified as the best-fit solution. The full accepted-model ensemble characterizes the range of velocity structures consistent with the observed dispersion curve under the adopted parameterization, search bounds, and structural prior.

---

## Pretrained Models

Two pretrained PADIT models are provided.

| Model               | File                      | Structural constraint                            |
| ------------------- | ------------------------- | ------------------------------------------------ |
| Monotonic PADIT     | `PADIT_monotonic.pth`     | Favors shear-wave velocity increasing with depth |
| LVZ-inclusive PADIT | `PADIT_LVZ_inclusive.pth` | Allows low-velocity zones and velocity reversals |

The **monotonic model** is suitable when velocity reversals are not expected or when a stronger monotonic structural constraint is preferred.

The **LVZ-inclusive model** should be considered when low-velocity zones or velocity reversals may be present.

The pretrained model files are distributed through the GitHub **Releases** page rather than stored directly in the source-code repository.

[Download pretrained models from GitHub Releases](https://github.com/CTJ-UT/U-SWIFT-A-Unified-Surface-Wave-Inversion-Framework-with-Transformer/releases)

After downloading, place the model files in the `models/` directory:

```text
models/
├── PADIT_monotonic.pth
└── PADIT_LVZ_inclusive.pth
```

---

## Quick Start

### 1. Download the Repository

Download the repository using **Code → Download ZIP**, or clone it with Git:

```bash
git clone https://github.com/CTJ-UT/U-SWIFT-A-Unified-Surface-Wave-Inversion-Framework-with-Transformer.git
cd U-SWIFT-A-Unified-Surface-Wave-Inversion-Framework-with-Transformer
```

---

### 2. Download the Pretrained Models

Download

```text
PADIT_monotonic.pth
PADIT_LVZ_inclusive.pth
```

from the latest GitHub Release and place them inside

```text
models/
```

---

### 3. Install Dependencies

Install the required Python packages using

```bash
pip install -r requirements.txt
```

---

### 4. Prepare the Dispersion Curve

An example input dispersion curve is provided in the `examples/` directory.

The input file contains three columns:

```text
frequency (Hz)    phase velocity (m/s)    standard deviation (m/s)
```

The data can be loaded using

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

If measurement uncertainties are not directly available, users should define `vr_std` according to the uncertainty adopted for their dispersion measurements.

---

### 5. Select the PADIT Model

The example notebook uses the monotonic model by default:

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

The notebook performs:

1. loading and sorting of the observed dispersion curve;
2. definition of the $z_{hs}$ and $V_{S,hs}$ search ranges;
3. dimensionless normalization;
4. PADIT inference;
5. denormalization;
6. forward modeling;
7. misfit evaluation;
8. identification of the best-fit model and accepted-model ensemble;
9. visualization of the inversion results.

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

The number of sampled $z_{hs}$ and $V_{S,hs}$ values can also be modified:

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

* `nd` controls the number of sampled half-space depths;
* `nv` controls the number of sampled half-space velocities.

Increasing `nd` and `nv` provides denser coverage of the search space but also increases computational cost.

If no accepted models are obtained, the prescribed search ranges should be reconsidered or broadened.

---

## Output

The inversion workflow produces:

* candidate $V_S$ profiles for the sampled $(z_{hs}, V_{S,hs})$ pairs;
* corresponding forward-modeled dispersion curves;
* misfit values for all candidate models;
* the best-fit $V_S$ profile;
* the accepted-model ensemble satisfying the prescribed misfit threshold;
* the corresponding accepted theoretical dispersion curves.

The accepted-model ensemble can be used to examine inversion nonuniqueness and the effect of the structural prior represented by the selected PADIT model.

---

## Choosing Between the Two PADIT Models

The two pretrained models encode different structural assumptions.

### Monotonic PADIT

Use

```text
PADIT_monotonic.pth
```

when an overall increase in $V_S$ with depth is considered a reasonable structural constraint.

This model generally provides a more restricted admissible solution space.

### LVZ-Inclusive PADIT

Use

```text
PADIT_LVZ_inclusive.pth
```

when low-velocity zones or velocity reversals should be considered.

Because this model represents a broader range of velocity structures, it may also produce a broader accepted-model ensemble.

The two models should therefore be viewed as representing different structural priors rather than as competing models with one being universally preferable.

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

### Main Files

**`main.ipynb`**
Example notebook implementing the complete SiDLIF inversion workflow.

**`src/aggregation_model.py`**
Implementation of the PADIT neural-network architecture.

**`src/utils.py`**
Utility functions for dispersion-curve normalization, resampling, neural-network prediction, denormalization, forward modeling, misfit calculation, and result processing.

**`examples/example_01.txt`**
Example fundamental-mode Rayleigh-wave dispersion curve.

**`models/`**
Local directory for pretrained PADIT weights downloaded from the GitHub Releases page.

---

## Current Scope and Assumptions

The current pretrained PADIT models were developed and validated for:

* fundamental-mode Rayleigh-wave dispersion curves;
* one-dimensional horizontally layered subsurface structures;
* 100 normalized layers above the half-space;
* a fixed Poisson's ratio of $1/3$ in the synthetic training models;
* constant density in the synthetic training models.

The present implementation focuses on inversion for $V_S$.

The 100-layer numerical discretization should not be interpreted as the intrinsic geophysical resolution of the inversion. Actual resolution is controlled primarily by the information content and frequency coverage of the observed dispersion curve.

Future extensions may incorporate additional elastic parameters, higher-mode dispersion information, or other geophysical observations.

---

## Citation

If you use SiDLIF in your research, please cite:

**Tianjian Cheng, Hongrui Xu, Jiayu Feng, Qiaomu Qi, Xiongyu Hu, and Chaofan Yao.
“SiDLIF: A Scale-Invariant Deep Learning-Based Inversion Framework for Surface-Wave Dispersion Curves via Physics-Guided Normalization.”**

Complete bibliographic information and the publication DOI will be added after publication.

---

## Open Research

The source code is maintained in this repository.

Pretrained PADIT model weights are distributed through GitHub Releases.

A versioned research archive will also be maintained on Zenodo.

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

or

**Hongrui Xu**
Faculty of Geosciences and Engineering
Southwest Jiaotong University
