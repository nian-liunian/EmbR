# Examples

This directory contains example configurations for different EmbR
workflows.


## Training example

The training example demonstrates how to train an EmbR model using a
prepared dataset.

Configuration:

    examples/train_manifest.json

Run:

``` bash
python run.py --manifest examples/train_manifest.json
```

The output includes the trained model checkpoint (`.ckpt`) and training
log (`loss.out`).

------------------------------------------------------------------------

## SCF-only example

The SCF-only example applies an existing trained EmbR model to perform
self-consistent QM/MM calculations.

Configuration:

    examples/scf_only_manifest.json

Run:

``` bash
python run.py --manifest examples/scf_only_manifest.json
```

Required inputs include: - a trained EmbR checkpoint; - a precomputed
NPZ cache.

------------------------------------------------------------------------

## Train directly from NPZ

For users who already have a precomputed NPZ cache, EmbR can be trained
directly without additional preprocessing.

Example:

``` bash
python train_soap_e0_mix_mmh.py \
    --soap-cache path/to/data.npz \
    --ckpt model.ckpt \
    --lr 6e-5 \
    --epochs 2000
```

------------------------------------------------------------------------

## Manuscript workflow configuration

The root-level file:

    manifest_mix.json

provides a multi-dataset configuration for the workflow described in
this work.

It can be executed with:

``` bash
python run.py --manifest manifest_mix.json
```
