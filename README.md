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

The dataset is available for downlading in a kaggle competition namely [RANZCR CLiP - Catheter and Line Position Challenge](https://www.kaggle.com/competitions/ranzcr-clip-catheter-line-classification/data).

## Data Preprocessing

There night be some mislabeled images so run `utils/preprocess_data.py` to relabel them.

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

If `train_fold5.csv` file is not available or if the number of folds to split data into is not 5 (Ex. 3, 4, 10...), run `utils/fold_split.py` to split data into folds:

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

Run `train_stage_1.py` to train teacher model (training customization can be done by modifying the configuration inside the file):

```
python train_stage_1.py
```

### Stage 2: Student model training

Run `train_stage_2_3.py` to train student model (training customization can be done by modifying the configuration inside the file):

```
python train_stage_2_3.py
```

**Note:** If pretrained weight is not available for downloading from timm by setting `pretrained=True`, try downloading it directly using link provided in timm repo then set `checkpoint_file=<downloaded-weight-file-path>` to load the weight. In case timm model fails to load due to the different format that the pretrained weight might have, run `utils/preprocess_checkpoint.py` (this will only work when timm provide the `checkpoint_filter_fn` in their implementation for the model specified).

For instance, to train `convnext_tiny`, go to https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py and find the corresponding checkpoint link (Ex. https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_1k_384.pth) provided inside the file then download into current directory and run `utils/preprocess_checkpoint.py` so a new checkpoint file with `_altered` at the end of the file name will be created.

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

**Note:** An alternative to run all the commands provided is to run the `train.sh` file

```
sh train.sh
```

## Demo

Run `utils/cam_vis.py` to demo classification result with GradCAM:

```
python utils/cam_vis.py
```

## Inference
For inferencing, run `inference.py` to generate predictions on the test set and the output file will be `submission.csv`:

```
python inference.py
```

**Note:** Depend on which checkpoint/model to use for inference, there might be need for modification of checkpoint/model directory inside the `inference.py` file to get the corresponding predictions. 