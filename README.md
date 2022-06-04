# kaggle-RANZCR-CliP

## Installation

```
git clone https://github.com/trantuan4132/kaggle-RANZCR-CliP
cd kaggle-RANZCR-CliP
```

## Data
The dataset is available for downlading in a kaggle competition namely [RANZCR CLiP - Catheter and Line Position Challenge](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification).

## Fold Splitting (Optional)

If `train_fold5.csv` file is not available or if the number of folds to split data into is not 5 (Ex. 3, 4, 10...), run `fold_split.py` to split data into folds

```
python fold_split.py

usage: fold_split.py [-h] [--file_path FILE_PATH] [--kfold KFOLD]

optional arguments:
  -h, --help            show this help message and exit
  --file_path FILE_PATH
  --kfold KFOLD
```

## Training

For training, run `train.py` to train model (training customization can be done by modifying the configuration inside the file)

```
python train.py
```

**Note:** If pretrained weight is not available for downloading from timm by setting `pretrained=True`, try downloading it directly using link provided in timm repo then set `checkpoint_file=<downloaded-weight-file-path>` to load the weight. In case timm model fails to load due to the different format that the pretrained weight might have, run `preprocess_checkpoint.py` (this will only work when timm provide the `checkpoint_filter_fn` in their implementation for the model specified)

```
python preprocess_checkpoint.py

usage: preprocess_checkpoint.py [-h] [--checkpoint_path CHECKPOINT_PATH] [--model MODEL] [--variant VARIANT]

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint_path CHECKPOINT_PATH
  --model MODEL
  --variant VARIANT
```

## Inference
For inferencing, run `inference.py` to generate predictions on the test set and the output file will be `submission.csv`

```
python inference.py
```