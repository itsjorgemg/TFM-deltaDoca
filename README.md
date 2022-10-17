# TFM-deltaDoca
Cybersecurity Master's Thesis - Evaluation of δ-DOCA on online machine learning

## How to run the project
1. Install the package manager pipenv.
```
pip install pipenv
```
2. Install dependencies.
```
pipenv install
```
The framework has been tested with Python 3.9, but higher versions will probably work too. Use `pipenv install --python path/to/python` to force a different version.

3. Spawn a virtual environment shell and run the code.
```
pipenv shell
python ./classification.py --help
```
Or alternatively:
```
pipenv run python ./classification.py --help
```

## Usage
```
usage: classification.py [-h] [--dataset DATASET_TRAIN] [--test DATASET_TEST] [--algorithm ALGORITHM] [--eps EPS] [--skip-original | --no-skip-original]

Run the analysis on the dataset

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET_TRAIN
                        path to the dataset to use for training
  --test DATASET_TEST   path to the dataset to use for testing
  --algorithm ALGORITHM
                        anonymization algorithm (only "doca" is available)
  --eps EPS             epsilon parameter for the doca algorithm
  --skip-original, --no-skip-original
                        skip analysis of original dataset (default: False)
```
