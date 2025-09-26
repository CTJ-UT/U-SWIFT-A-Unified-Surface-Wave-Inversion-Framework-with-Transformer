# U-SWIFT: A Unified Surface Wave Inversion Framework with Transformer

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

**U-SWIFT is a unified deep learning framework for inverting surface wave dispersion curves to obtain shear wave velocity (Vs) profiles.**

This repository contains the official implementation of the paper: "[U-SWIFT: A Unified Surface Wave Inversion Framework with Transformer via Normalization of Dispersion Curves]".

The key innovation of this framework is the normalization of dispersion curves based on their scaling properties. [cite_start]This allows a single, pre-trained model to robustly predict Vs profiles across diverse depth and velocity scales, from near-surface engineering applications to deep crustal imaging, without retraining. [cite: 5, 22, 35, 36, 39, 703]
