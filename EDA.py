import pandas as pd


# 1. Load & preprocess
def clean_data(filepath='train_subset.csv'):

    df = pd.read_csv(filepath)


    df_clean = df[
        (df['ap_hi'] >= 80) & (df['ap_hi'] <= 200) &
        (df['ap_lo'] >= 50) & (df['ap_lo'] <= 130)
    ].copy()


    df_clean['cardio_C'] = df_clean['cardio'].map({'No': 0, 'Yes': 1})

    for col in ['smoke', 'alco', 'active']:
        df_clean[col] = df_clean[col].map({'No': 0, 'Yes': 1})

    return df_clean



if __name__ == '__main__':
    raw = pd.read_csv('train_subset.csv')
    print("=== RAW DATA SUMMARY (before filtering) ===")
    print(raw.describe(), "\n")


    clean_df = clean_data()
    print(clean_df[['ap_hi','ap_lo']].describe())
    print(clean_df.isnull().sum())
    print(clean_df.describe())

