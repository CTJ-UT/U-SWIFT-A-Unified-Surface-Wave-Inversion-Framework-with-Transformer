# SiDLIF: A Scale-Invariant Deep Learning-Based Inversion Framework for Surface-Wave Dispersion Curves via Physics-Guided Normalization

SiDLIF is a physics-guided deep learning framework for inverting **fundamental-mode Rayleigh-wave dispersion curves** into shear-wave velocity ((V_S)) profiles.

The framework exploits the scaling properties of surface-wave dispersion to transform dispersion curves and corresponding velocity profiles into **dimensionless representations**. This decouples the learned inverse mapping from absolute depth and velocity scales, allowing the same pretrained model to be applied to problems ranging from shallow near-surface investigations to crustal-scale imaging without site-specific retraining.

This repository provides the implementation of the paper:

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

  * a **monotonic model**, which favors (V_S) increasing with depth;
  * an **LVZ-inclusive model**, which allows low-velocity zones and velocity reversals.

* **Forward-model-constrained inversion**
  Candidate (V_S) profiles predicted by PADIT are validated by forward modeling of their theoretical dispersion curves.

* **Accepted-model ensembles**
  Solutions satisfying a prescribed dispersion-curve misfit threshold are retained as an ensemble of physically admissible models, providing a practical representation of inversion nonuniqueness.

* **Minimal manual parameterization**
  Users do not need to prescribe a site-specific number of layers. The primary search parameters are the half-space depth (z_{hs}) and half-space shear-wave velocity (V_{S,hs}).

---

## Method Overview

### 1. Dimensionless normalization

For a subsurface model with half-space depth (z_{hs}) and half-space shear-wave velocity (V_{S,hs}), the velocity profile is normalized as

[
z^*=\frac{z}{z_{hs}},
]

[
V_S^*=\frac{V_S}{V_{S,hs}}.
]

The corresponding dispersion curve is normalized as

[
c^*=\frac{c}{V_{S,hs}},
]

[
f^*=\frac{fz_{hs}}{V_{S,hs}},
]

where (f) is frequency and (c) is Rayleigh-wave phase velocity.

The resulting normalized profile has

[
z_{hs}^*=1,
\qquad
V_{S,hs}^*=1.
]

This representation allows PADIT to learn the relationship between dispersion curves and velocity structures independently of their absolute physical scales.

### 2. Grid search over half-space parameters

Because (z_{hs}) and (V_{S,hs}) are generally unknown, SiDLIF evaluates multiple candidate pairs over user-defined search ranges.

In the example notebook, the initial ranges are estimated from the maximum resolved wavelength,

[
\lambda_{\max}=\max\left(\frac{c}{f}\right),
]

using

[
z_{hs}\in[0.2\lambda_{\max},,0.7\lambda_{\max}],
]

and

[
V_{S,hs}\in[1.05c_{\max},,3c_{\max}].
]

These intervals are intended as practical initial search ranges and can be modified when additional prior information is available.

The default example samples 200 values for (z_{hs}) and 200 values for (V_{S,hs}), producing 40,000 candidate parameter pairs.

### 3. PADIT inversion

Each normalized dispersion curve is passed to the **Point-wise Additive Dispersion Inversion Transformer (PADIT)**.

The network consists of:

* a point-wise feature extractor;
* positional encoding;
* six pre-LayerNorm Transformer encoder layers;
* sum aggregation over the sequence dimension;
* a regression head producing 100 normalized (V_S^*) values;
* a Sigmoid output activation.

The output represents 100 equally spaced layers above the half-space, with normalized layer thickness

[
t^*=0.01.
]

### 4. Denormalization

The predicted dimensionless velocity profile is transformed back to the original physical scale using

[
z=z^*z_{hs},
]

[
V_S=V_S^*V_{S,hs}.
]

Each candidate ((z_{hs},V_{S,hs})) pair therefore produces one candidate physical (V_S) profile.

### 5. Forward-model validation

For every candidate (V_S) profile, a theoretical fundamental-mode Rayleigh-wave dispersion curve is calculated.

The misfit is defined as

[
\mathrm{misfit}
===============

\sqrt{
\frac{1}{m}
\sum_{j=1}^{m}
\left(
\frac{c_j^{obs}-c_j^{theo}}
{\sigma_j}
\right)^2
},
]

where

* (c_j^{obs}) is the observed phase velocity,
* (c_j^{theo}) is the theoretical phase velocity,
* (\sigma_j) is the observational uncertainty,
* (m) is the number of dispersion-curve points.

Candidate profiles with

[
\mathrm{misfit}<1
]

are retained as **accepted models**.

The model with the lowest misfit is the best-fit solution, while the accepted-model ensemble represents the range of structures consistent with the observed dispersion curve under the adopted parameterization and structural prior.

---

## Pretrained Models

Two pretrained PADIT models are provided with the SiDLIF release.

| Model               | File                      | Structural prior                                 |
| ------------------- | ------------------------- | ------------------------------------------------ |
| Monotonic PADIT     | `PADIT_monotonic.pth`     | (V_S) generally increases with depth             |
| LVZ-inclusive PADIT | `PADIT_LVZ_inclusive.pth` | Allows low-velocity zones and velocity reversals |

The **monotonic model** is recommended when velocity reversals are not expected or when a stronger structural constraint is desired.

The **LVZ-inclusive model** is recommended when low-velocity zones or velocity reversals should be considered.

The pretrained `.pth` files are distributed through the GitHub **Releases** page rather than stored directly in the source-code repository:

https://github.com/CTJ-UT/U-SWIFT-A-Unified-Surface-Wave-Inversion-Framework-with-Transformer/releases

After downloading, place the model files in

```text
models/
├── PADIT_monotonic.pth
└── PADIT_LVZ_inclusive.pth
```

---

## Quick Start

### 1. Download the repository

Download the source code from GitHub using **Code → Download ZIP**, or clone the repository:

```bash
git clone https://github.com/CTJ-UT/U-SWIFT-A-Unified-Surface-Wave-Inversion-Framework-with-Transformer.git
cd U-SWIFT-A-Unified-Surface-Wave-Inversion-Framework-with-Transformer
```

### 2. Download the pretrained models

Download the following files from the latest SiDLIF release:

```text
PADIT_monotonic.pth
PADIT_LVZ_inclusive.pth
```

Place them inside the `models/` directory.

### 3. Install dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The main dependencies include:

* PyTorch
* NumPy
* SciPy
* scikit-learn
* disba
* Matplotlib
* tqdm
* joblib

### 4. Prepare the dispersion curve

The example input file is

```text
examples/example_01.txt
```

Each row contains three columns:

```text
frequency(Hz)    phase_velocity(m/s)    standard_deviation(m/s)
```

For example:

```text
20.0    363.379823    7.267596
22.0    313.394214    6.267884
24.0    272.048227    5.440965
```

The example notebook reads the data using

```python
data = np.loadtxt("examples/example_01.txt")

f = data[:, 0]
vr = data[:, 1]
vr_std = data[:, 2]
```

If observational standard deviations are not available, users may define `vr_std` directly according to the uncertainty adopted for their dispersion measurements.

### 5. Select the PADIT model

The default example uses the monotonic model:

```python
model_path = "models/PADIT_monotonic.pth"
```

To use the LVZ-inclusive model, replace it with

```python
model_path = "models/PADIT_LVZ_inclusive.pth"
```

### 6. Run the inversion

Open

```text
main.ipynb
```

and execute the cells sequentially.

The notebook performs:

1. dispersion-data loading and sorting;
2. estimation of (z_{hs}) and (V_{S,hs}) search ranges;
3. dimensionless normalization;
4. PADIT inference;
5. denormalization;
6. forward computation;
7. misfit evaluation;
8. visualization of the accepted dispersion curves and (V_S) profiles.

---

## Adjusting the Search Space

The default search intervals in `main.ipynb` are

```python
vs_bounds_halfspace = np.array([
    1.05 * np.max(vr),
    3.0 * np.max(vr)
])

depth_bounds_halfspace = np.array([
    0.2,
    0.7
]) * np.max(vr / f)
```

The number of sampled values can also be changed:

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

Increasing `nd` and `nv` provides denser coverage of the (z_{hs})-(V_{S,hs}) parameter space but increases computational cost.

If no accepted models are obtained, the search ranges should be reconsidered or broadened.

---

## Output

The inversion produces:

* candidate (V_S) profiles for all sampled ((z_{hs},V_{S,hs})) pairs;
* corresponding forward-modeled dispersion curves;
* dispersion-curve misfit values;
* the best-fit (V_S) profile;
* an ensemble of accepted (V_S) profiles satisfying the misfit threshold;
* the corresponding accepted theoretical dispersion curves.

The accepted-model ensemble can be used to examine the nonuniqueness of the inversion and the effect of different structural priors.

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

### Main files

`main.ipynb`
Example workflow for running a complete SiDLIF inversion.

`src/aggregation_model.py`
Implementation of the PADIT neural-network architecture.

`src/utils.py`
Functions for normalization, resampling, neural-network prediction, denormalization, forward modeling, misfit calculation, and inversion-result processing.

`examples/example_01.txt`
Example fundamental-mode Rayleigh-wave dispersion curve.

`models/`
Local directory for pretrained PADIT weights downloaded from the GitHub Releases page.

---

## Current Scope and Assumptions

The current pretrained PADIT models were developed and validated for:

* **fundamental-mode Rayleigh-wave dispersion curves**;
* one-dimensional horizontally layered subsurface structures;
* 100 normalized layers above the half-space;
* a fixed Poisson's ratio of (1/3) in the synthetic training models;
* constant density in the synthetic training models.

The present implementation focuses on (V_S) inversion. Extensions incorporating additional elastic parameters, higher-mode dispersion information, or other geophysical observations are possible directions for future development.

The 100-layer numerical representation should not be interpreted as the intrinsic geophysical resolution of the inversion. Actual resolution depends on the information content and frequency coverage of the observed dispersion curve.

---

## Citation

If you use SiDLIF in your research, please cite the associated manuscript:

**Tianjian Cheng, Hongrui Xu, Jiayu Feng, Qiaomu Qi, Xiongyu Hu, and Chaofan Yao.
“SiDLIF: A Scale-Invariant Deep Learning-Based Inversion Framework for Surface-Wave Dispersion Curves via Physics-Guided Normalization.”**

The complete bibliographic information and DOI will be added after publication.

---

## License

This project is distributed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

See the `LICENSE` file for details.

---

## Contact

For questions regarding the method or implementation, please contact:

**Tianjian Cheng**
Faculty of Geosciences and Engineering
Southwest Jiaotong University

or

**Hongrui Xu**
Faculty of Geosciences and Engineering
Southwest Jiaotong University
