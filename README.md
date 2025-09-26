# U-SWIFT: A Unified Surface Wave Inversion Framework with Transformer

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

**U-SWIFT is a unified deep learning framework for inverting surface wave dispersion curves to obtain shear wave velocity (Vs) profiles.**

This repository contains the official implementation of the paper: "[U-SWIFT: A Unified Surface Wave Inversion Framework with Transformer via Normalization of Dispersion Curves]".

The key innovation of this framework is the normalization of dispersion curves based on their scaling properties. [cite_start]This allows a single, pre-trained model to robustly predict Vs profiles across diverse depth and velocity scales, from near-surface engineering applications to deep crustal imaging, without retraining.

## ✨ Key Features

**Scale Invariance**: By leveraging the normalization of dispersion curves based on their physical scaling properties, a single pre-trained model can be applied to inversion problems at vastly different scales—from shallow near-surface (<10m) to deep crustal levels—without any retraining.
**Variable-Length Input**: The framework incorporates a Transformer-based model (PADIT) that is specifically designed to process dispersion curves of varying lengths, eliminating the need for fixed-size inputs.
**Robust Uncertainty Quantification**: The framework can rapidly generate an ensemble of valid Vs profiles that fit the observed data [cite: 28][cite_start], allowing for a robust and meaningful quantification of the inversion uncertainty.
**Simplified Workflow**: This approach eliminates the need for tedious manual parameterization (e.g., defining the number of layers)[cite: 738]. [cite_start]Users only need to provide a broad estimate for the half-space depth and velocity to obtain accurate results.
