# kaggle-RANZCR-CliP

## Installation

```
git clone https://github.com/trantuan4132/kaggle-RANZCR-CliP
cd kaggle-RANZCR-CliP
```

## Set up environment

```
pip install -r requirements.txt
```

## Data
The dataset is available for downlading in a kaggle competition namely [RANZCR CLiP - Catheter and Line Position Challenge](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification).

## Processing
There night be some mislabeled images so run `utils/preprocess_data.py` to relabel them
```
python utils/preprocess_data.py

usage: preprocess_data.py [-h] [--label_path LABEL_PATH] [--annot_path ANNOT_PATH] [--relabel_path RELABEL_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --label_path LABEL_PATH
                        Path to the label file
  --annot_path ANNOT_PATH
                        Path to the annotation file
  --relabel_path RELABEL_PATH
                        Path to the relabel file
```


## Fold Splitting (Optional)

If `train_fold5.csv` file is not available or if the number of folds to split data into is not 5 (Ex. 3, 4, 10...), run `utils/fold_split.py` to split data into folds

```
python utils/fold_split.py

usage: fold_split.py [-h] [--label_path LABEL_PATH] [--kfold KFOLD] [--annot_path ANNOT_PATH] [--save_path SAVE_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --label_path LABEL_PATH
                        Path to the label file
  --kfold KFOLD         Number of folds
  --annot_path ANNOT_PATH
                        Path to the annotation file
  --save_path SAVE_PATH
                        Path to the save file
```

## Training

### Stage 1: Teacher model training

For training, run `train_stage_1.py` to train model (training customization can be done by modifying the configuration inside the file)

```
python train_stage_1.py
```

**Note:** If pretrained weight is not available for downloading from timm by setting `pretrained=True`, try downloading it directly using link provided in timm repo then set `checkpoint_file=<downloaded-weight-file-path>` to load the weight. In case timm model fails to load due to the different format that the pretrained weight might have, run `utils/preprocess_checkpoint.py` (this will only work when timm provide the `checkpoint_filter_fn` in their implementation for the model specified)

```
python utils/preprocess_checkpoint.py

usage: preprocess_checkpoint.py [-h] [--checkpoint_path CHECKPOINT_PATH] [--model MODEL] [--variant VARIANT]

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint_path CHECKPOINT_PATH
                        Path to the checkpoint file
  --model MODEL         Model name
  --variant VARIANT     Model variant
```

<!-- ## Inference
For inferencing, run `inference.py` to generate predictions on the test set and the output file will be `submission.csv`

```
python inference.py
``` -->