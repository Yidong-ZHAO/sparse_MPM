# Sparse MPM

Sparse material point method for efficient large-scale simulation.
This repository is the GPU implementation based on [NVIDIA Warp](https://nvidia.github.io/warp/).


The code is associated with the preprint:

**Unified sparse framework for large-scale material point method simulations**

arXiv: https://arxiv.org/abs/2605.28525


## Getting started
### Installation
```bash
pip install -r requirements.txt
```

### How to run
Navigate to an example folder and execute the corresponding Python script:
```
cd examples/mountain/
python mountain_sparse.py
```

## Code structure
Simulations are organized into two main directories:
- `examples/` contains individual simulation scripts specifying setups.
- `utilities/` includes other utility functions such as timer.

In each example foler, we provide a single, standalone file to help new users quickly understand the structure without cross-referencing multiple files.