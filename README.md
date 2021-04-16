# Invertible deep priors

## Prerequisites

First clone the repository:

```bash
git clone https://github.com/alisiahkoohi/invertible-deep-prior
cd invertible-deep-prior/
```

This software is based on [PyTorch-1.8.1](https://github.com/pytorch/pytorch/releases/tag/v1.8.0) and [FrEIA-0.2](https://github.com/VLL-HD/FrEIA/releases/tag/v0.2).

Follow the steps below to install the necessary libraries. If you have a CUDA-enabled GPU, run:

```bash
conda env create -f environment.yml
source activate freia
```

You may want to specify your CUDA Toolkit version in the `environment.yml` file above. If you don't have a GPU, run:


```bash
conda env create -f environment-cpu.yml
source activate freia-cpu
```

## Example

To run the example, execute:

```bash
python src/main.py --cuda 1 --lr 0.0001 --max_itr 5001
```

## Author

Ali Siahkoohi
