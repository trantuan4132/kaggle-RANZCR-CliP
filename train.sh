python utils/preprocess_data.py --drop
python utils/fold_split.py
python utils/preprocess_checkpoint.py
python train_stage_1.py
python train_stage_2_3.py