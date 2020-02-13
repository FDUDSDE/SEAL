# SEAL
Codes and data for KDD2020 submission.

## Usage

Un-tar processed datasets:

```
chmod +x untar_files.sh
./untar_files.sh
```

Or you could run pre-processing scripts in `preprocessing/` folder to obtain datasets from raw data.


Run the algorithm:

```
python run_seal_attr.py --dataset dblp --locator_coef 0.5 --radius_penalty 0.0
```

## Dependencies

All dependencies are included in `env.yml`.

You could install all dependencies with `conda`:

```
conda env create -f env.yml
```