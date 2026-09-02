# EmbR

Physically constrained machine-learning repulsion embedding for QM/MM
electronic-structure calculations.

## About The Project

Conventional QM/MM methods describe the molecular environment using
fixed point charges. Although computationally efficient, this
representation lacks short-range electronic effects arising from the
response of the surrounding environment.

EmbR introduces a machine-learning-based repulsive embedding correction
that restores missing short-range electronic interactions while
retaining the efficiency of QM/MM calculations.

The EmbR workflow consists of:

-   generating full_QM reference interaction energies;
-   constructing SOAP-based local environment descriptors;
-   training a machine-learning repulsion model;
-   applying the learned correction through self-consistent QM/MM
    calculations.

`dE` corresponds to the interaction-energy difference ΔE used in the
manuscript.

------------------------------------------------------------------------

# Built With

## Electronic structure

  Component                Software
  ------------------------ -------------
  QM calculations          PySCF
  QM/MM embedding          PySCF QM/MM
  Reference calculations   HF/full_QM

## Machine learning

  Component    Software
  ------------ ----------
  Descriptor   SOAP
  Framework    PyTorch
  Model        EmbR

------------------------------------------------------------------------

# Getting Started

## Installation

Install the required environment:

``` bash
conda create -n embr python=3.11
conda activate embr
pip install -r requirements.txt
```

------------------------------------------------------------------------

# Usage

## Quick Start

A minimal example can be used to quickly test the EmbR workflow.

Create:

    examples/mini.json

with:

``` json
{
  "datasets": [
    {
      "mode": "ab_initio",
      "prefix": "Gly+",
      "out_dir": "examples",
      "QM_atoms": 10,
      "qm_charge": 0,
      "n_frames": 10,
      "e0_file": "EGly+.txt"
    }
  ],
  "train": {
    "lr": 6e-5
  }
}
```

Run:

``` bash
python run.py --manifest examples/mini.json
```

The workflow will automatically use default settings for unspecified
options.

Generated files are described in the **Output files** section.

------------------------------------------------------------------------

## Main workflow: run.py

`run.py` is the main interface for EmbR calculations. All computational
settings are controlled through a JSON manifest file.

For example:

``` bash
python run.py --manifest manifest_mix.json
```

The provided `manifest_mix.json` contains a complete workflow
configuration including reference calculation, preprocessing, training,
and EmbR calculations.

For dataset description and pretrained model availability, please refer to [datasets.md](datasets.md).


------------------------------------------------------------------------

# Calculation Modes

The main calculation modes are:

## ab_initio

Generates reference data and prepares training resources.

Required input:

    out_dir/
    └── Coo/
        ├── Coo1.xyz
        ├── Coo2.xyz
        └── ...

The coordinate files should contain QM atoms followed by MM atoms.

MM charges can either: 
- be provided directly in the Coo.xyz by yourself (Append the corresponding charge after the z-coordinate);
- be generated using `charge_mode`.

------------------------------------------------------------------------

## train

Trains an EmbR model from prepared training data.

Required input:

-   reference interaction energies and electron densities.

Output:

-   trained checkpoint (`.ckpt`).

------------------------------------------------------------------------

## train_with_npz

Directly trains an EmbR model from an existing precomputed `.npz` file.

Required input:

-   SOAP-cache `.npz`.


------------------------------------------------------------------------

## scf

Applies a trained EmbR model in self-consistent QM/MM calculations.

Required input:

-   trained checkpoint;
-   precomputed soap-cache.

------------------------------------------------------------------------

## Predict dE from a trained checkpoint

A trained EmbR checkpoint can be used to predict the energy correction (dE, corresponding to ΔE in the manuscript) for new structures.

There are two supported workflows.

### 1. Prediction from an existing SOAP cache

If the SOAP cache (`.npz`) has already been generated, prediction can be performed directly:

```bash
python predict_delta_e.py \
    --ckpt path/to/model.ckpt \
    --soap-npz path/to/cache.npz \
    -o predicted_dE.txt
```
The output file contains the predicted dE values for each structure in the input cache.

If you want to compare your prediction with full-QM, add

```bash
--e0-file path/to/E0.txt \
```

Output file will contain columns for full-QM and predict, and it will also output the root mean square error (RMSE).

### 2. Prediction starting from Coo.xyz structures

For new molecular configurations provided as Coo*.xyz, EmbR can automatically generate the required SOAP features and apply the trained checkpoint.

Required:
-   trained checkpoint;
-   a directory containing Coo*.xyz structures
-   the corresponding reference embedding information (ref directory); if do not have reference embedding information, run `ab_initio` mode to get them

Example:
```bash
python predict_delta_e.py \
    --ckpt path/to/model.ckpt \
    --coo-dir path/to/Coo \
    --ref-dir path/to/ref \
    --qm-atoms 10 \
    --i0 1 \
    --n-frames 10 \
    -o predicted_dE.txt
```
The script automatically performs the required feature generation and then predicts dE values for the provided structures.

The prediction range can be controlled using `--i0` and `--n-frames` when only a subset of Coo structures needs to be evaluated.

# Output Files

Typical output files include:

  File       Description
  ---------- ------------------------------------------
  `ref/`     full_QM reference data
  
  `*.npz`    precomputed SOAP-cache data
  
  `*.ckpt`   trained EmbR model checkpoint
  
  `scf/`     EmbR self-consistent calculation results
  

------------------------------------------------------------------------

# Examples

Example manifest files are provided in the repository:

  File                                Description
  ----------------------------------- --------------------------------------
  `manifest.json`                     Single-dataset configuration
  
  `manifest_mix.json`                 Mixed-system workflow configuration
  
  `examples/train_manifest.json`      Train_with_npz example
  
  `examples/scf_only_manifest.json`   SCF-only example

Example commands:

Complete workflow:

``` bash
python run.py --manifest examples/all_manifest.json
```

Train_with_npz:

``` bash
python run.py --manifest examples/train_manifest.json
```

SCF calculation:

``` bash
python run.py --manifest examples/scf_only_manifest.json
```

------------------------------------------------------------------------

# Parameters

Detailed parameter descriptions are provided in:

    PARAMETERS.txt

The file describes all available manifest options, including:

-   dataset settings;
-   embedding and calculation settings;
-   SOAP descriptor settings;
-   training settings.

------------------------------------------------------------------------

# Contributing

Issues and pull requests are welcome.

------------------------------------------------------------------------

# License

See the LICENSE file for details.
