import argparse
from sklearn.model_selection import GroupKFold
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str, default='train.csv',
                        help='Path to the label file')
    parser.add_argument('--kfold', type=int, default=5,
                        help='Number of folds')
    parser.add_argument('--annot_path', type=str, default='train_annotations.csv',
                        help='Path to the annotation file')
    parser.add_argument('--save_path', type=str, default='train_fold5.csv',
                        help='Path to the save file')
    return parser.parse_args()


def fold_split(df_path, label_cols, kfold, df_annot_path='', save_path=''):
    """
    Split the dataset into kfold

    Args:
    -----
    df_path: str
        Path to the label file
    label_cols: list
        List of label columns
    kfold: int
        Number of folds
    df_annot_path: str, optional
        Path to the annotation file
    save_path: str, optional
        Path to the save file
    """
    df = pd.read_csv(df_path)
    combined_label = pd.Series(df[label_cols].astype('str').values.sum(axis=1))
    if df_annot_path:
        df_annot = pd.read_csv(df_annot_path)
        has_annot = pd.Series('0', index=df['StudyInstanceUID'])
        has_annot.loc[df_annot['StudyInstanceUID'].unique()] = '1'
        combined_label = has_annot.reset_index(drop=True) + combined_label
        # df['has_annot'] = has_annot.reset_index(drop=True).astype('int')
    combined_label = combined_label.apply(lambda x: int(x, 2))
    df['fold'] = -1
    gkf = GroupKFold(n_splits=kfold)
    for fold, (_, val_idx) in enumerate(gkf.split(df.index, 
                                                  y=combined_label,
                                                  groups=df['PatientID'])):
        df.loc[val_idx, 'fold'] = fold
    if save_path:
        df.to_csv(save_path, index=False)
    return df


if __name__ == '__main__':
    args = parse_args()
    label_cols = [
        'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal', 
        'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
        'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal', 
        'Swan Ganz Catheter Present',
    ]
    # save_path = args.label_path.replace('.csv', f'_fold{args.kfold}.csv')
    df = fold_split(args.label_path, label_cols, args.kfold, args.annot_path, args.save_path)
    print(df)

    # # Plot fold distribution
    # plt.figure(figsize=(8, 6))
    # sns.countplot(x='fold', data=df)
    # plt.show()

    # # Plot fold distribution per label
    # # label_cols = label_cols + ['has_annot'] if args.annot_path else label_cols
    # label_cnt_per_fold = df.groupby('fold')[label_cols].sum().stack().reset_index()
    # label_cnt_per_fold.columns = ['fold', 'label', 'count']
    # plt.figure(figsize=(10, 6))
    # sns.barplot(x='count', y='label', hue='fold', 
    #             data=label_cnt_per_fold.sort_values('count', ascending=False))
    # plt.show()