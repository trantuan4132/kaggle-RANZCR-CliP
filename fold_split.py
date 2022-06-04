import argparse
from sklearn.model_selection import GroupKFold
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='train.csv')
    parser.add_argument('--kfold', type=int, default=5)
    return parser.parse_args()


def fold_split(df_path, label_cols, kfold):
    df = pd.read_csv(df_path)
    combined_label = df[label_cols].astype('str').values.sum(axis=1)
    combined_label = pd.Series(combined_label).apply(lambda x: int(x, 2))
    df['fold'] = -1
    gkf = GroupKFold(n_splits=kfold)
    for fold, (_, val_idx) in enumerate(gkf.split(df.index, 
                                                  y=combined_label,
                                                  groups=df['PatientID'])):
        df.loc[val_idx, 'fold'] = fold
    save_path = df_path.replace('.csv', f'_fold{kfold}.csv')
    df.to_csv(save_path, index=False)
    return df


def main():
    args = parse_args()
    label_cols = [
        'ETT - Abnormal', 'ETT - Borderline',
        'ETT - Normal', 'NGT - Abnormal', 'NGT - Borderline',
        'NGT - Incompletely Imaged', 'NGT - Normal', 'CVC - Abnormal',
        'CVC - Borderline', 'CVC - Normal','Swan Ganz Catheter Present'
    ]
    df = fold_split(args.file_path, label_cols, args.kfold)
    print(df)

    # # Plot fold distribution
    # plt.figure(figsize=(8, 6))
    # sns.countplot(x='fold', data=df)
    # plt.show()

    # # Plot fold distribution per label
    # label_cnt_per_fold = df.groupby('fold')[label_cols].sum().stack().reset_index()
    # label_cnt_per_fold.columns = ['fold', 'label', 'count']
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x='count', y='label', hue='fold', 
    #             data=label_cnt_per_fold.sort_values('count', ascending=False))
    # plt.show()

if __name__ == '__main__':
    main()