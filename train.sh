python utils/preprocess_data.py \
    --label_path "train.csv" \
    --annot_path "train_annotations.csv" \
    --relabel_path "relabel.csv" \
    --drop
python utils/fold_split.py \
    --label_path "train.csv" \
    --kfold 5 \
    --annot_path "train_annotations.csv" \
    --save_path "train_fold5.csv"
python utils/preprocess_checkpoint.py \
    --checkpoint_path "convnext_tiny_22k_1k_384.pth" \
    --model "convnext" \
    --variant "convnext_tiny"
python train_stage_1.py
python train_stage_2_3.py