# Datasets and Pretrained Models

The complete data package is available on Zenodo:

**Zenodo DOI:** \[https://doi.org/10.5281/zenodo.22239030\]

The archive contains the training datasets, test datasets, pretrained
models, and precomputed descriptor caches required for reproducing the
EmbR workflow.

------------------------------------------------------------------------

## Download and Setup

After downloading the data archive from Zenodo, place the archive in the
root directory of the EmbR repository. It includes an archive `data.backup.tar.gz` and two soap-caches `mix_+.npz`, `mix_600.npz`.


Extract the archive:

``` bash
tar -zxvf data.backup.tar.gz
```

The extracted directory should be:

    EmbR/
    └── data/
        ├── train/
        ├── test/
        ├── mix_+.ckpt
        └── mix_600.ckpt


------------------------------------------------------------------------

## Dataset Organization

### `train/`

The training dataset used for developing EmbR models.

The dataset contains molecular configurations generated using different
basis sets:

-   Datasets with `+` correspond to the diffuse basis set **6-31+G\***.
-   Datasets without `+` correspond to the standard basis set
    **6-31G\***.

The training data include:

-   `Gly`
-   `Gly+`
-   `Ala`
-   `Ala+`
-   `Asp`
-   `Asp+`
-   `Lys`
-   `Lys+`

------------------------------------------------------------------------

### `test/`

The independent test datasets used for evaluating the generalization
ability of trained EmbR models.

------------------------------------------------------------------------

## Pretrained EmbR Checkpoints

### `mix_+.ckpt`

Pretrained EmbR checkpoint trained using the **6-31+G\*** dataset.

### `mix_600.ckpt`

Pretrained EmbR checkpoint trained using the **6-31G\*** dataset.

Both checkpoints can be directly used for dE prediction.

------------------------------------------------------------------------

## Precomputed SOAP Caches

### `mix_+.npz`

SOAP cache generated from the **6-31+G\*** dataset.

This file contains the precomputed descriptors required for EmbR
training and can be used directly without repeating SOAP preprocessing.


### `mix_600.npz`

SOAP cache generated from the **6-31G\*** dataset.

This file can also be directly used as training input.

You can set the right path of `.npz` in the `example/train_manifest.json` and run:

```bash
python run.py --manifest example/train_manifest.json
```

to train.

------------------------------------------------------------------------

## Usage

Example prediction using a pretrained checkpoint:

``` bash
python predict_delta_e.py \
    --ckpt data/mix_+.ckpt \
    --soap-npz data/test/Ala+_test/Ala+_test.npz \
    --e0-file data/test/Ala+_test/EAla+_test.txt \
    -o predicted_dE.txt
```

