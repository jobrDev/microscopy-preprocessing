Microscopy Preprocessing

A collection of modular, general-purpose Python functions for preprocessing microscopy images.

This repository provides reusable image processing tools that can be applied across different microscopy modalities and experimental workflows. The aim is to offer a common set of preprocessing functions that can be easily integrated into custom analysis pipelines without being tied to a specific imaging technique.

Features

The repository currently includes the following preprocessing methods:

Background subtraction (background.py): Functions for estimating and removing background signal from microscopy images.
Edge damping (edge_damping.py): Functions for reducing edge-related artefacts and mitigating boundary effects in image processing.
Drift correction (drift.py): Functions for correcting spatial shifts and drift between image frames.
Intensity normalization (normalization.py): Functions for normalizing image intensities to improve consistency and comparability between images.
Design principles
Modularity: Each preprocessing technique is implemented in a separate module.
Reusability: Functions are designed to be independent of specific microscopy modalities or analysis pipelines.
Flexibility: Individual functions can be used independently or combined into custom preprocessing workflows.
Reproducibility: A shared implementation of preprocessing methods helps maintain consistency across different projects and analyses.
Installation

Clone the repository and install it in editable mode:

bash
git clone https://github.com/<user-or-org>/microscopy-preprocessing.git
cd microscopy-preprocessing
pip install -e .
Dependencies
NumPy
SciPy
scikit-image
Usage

Individual preprocessing functions can be imported from their respective modules and incorporated into custom image analysis workflows.

Scope

This repository focuses exclusively on general-purpose microscopy image preprocessing. It does not contain microscopy-modality-specific pipelines, ROI selection tools, or downstream image analysis methods.

Contributing

Contributions, improvements, and additional general-purpose preprocessing methods are welcome. The goal is to develop a shared collection of well-documented, reusable tools for microscopy image analysis.
