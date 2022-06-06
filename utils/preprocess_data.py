import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_path', type=str, default='train.csv',
                        help='Path to the label file')
    parser.add_argument('--annot_path', type=str, default='train_annotations.csv',
                        help='Path to the annotation file')
    parser.add_argument('--relabel_path', type=str, default='relabel.csv',
                        help='Path to the relabel file')
    return parser.parse_args()


def relabel(img_id, df, df_annot, img_col, annot_idx, old_label, new_label):
    df.loc[df[img_col]==img_id, old_label] = 0
    df.loc[df[img_col]==img_id, new_label] = 1
    df_annot.loc[annot_idx, 'label'] = new_label
    return df, df_annot


def clean_data(df_relabel, df, df_annot, img_col):
    # img_id = '1.2.826.0.1.3680043.8.498.93345761486297843389996628528592497280'
    for index, row in df_relabel.iterrows():
        df, df_annot = relabel(row[img_col], df, df_annot, img_col, annot_idx=row['index'], 
                               old_label=row['label'], new_label=row['new_label'])
    return df, df_annot


# Get data with annotation only
def get_annotated_data(df, df_annot, img_col, label_cols):
    annot_img_id = df_annot[img_col].unique()
    return df, df_annot


if __name__ == "__main__":
    args = parse_args()
    df = pd.read_csv(args.label_path)
    df_annot = pd.read_csv(args.annot_path)
    img_col = 'StudyInstanceUID'
    df_relabel = pd.read_csv(args.relabel_path, index_col=0)
    df, df_annot = clean_data(df_relabel, df, df_annot, img_col)
    # print(df.query(f"{img_col}=='1.2.826.0.1.3680043.8.498.93345761486297843389996628528592497280'"))
    # print(df_annot.query(f"{img_col}=='1.2.826.0.1.3680043.8.498.93345761486297843389996628528592497280'"))
    # df.to_csv(args.label_path, index=False)
    # df_annot.to_csv(args.annot_path, index=False)