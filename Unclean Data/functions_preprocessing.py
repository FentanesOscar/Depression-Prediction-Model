import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def show_invalid_entries(df, column, invalid_list, after=False):
    invalid_entries = df[df[column].isin(invalid_list)]
    if after:
        print(f"After cleaning — number of invalid {column} entries: {len(invalid_entries)}")
    else:
        print(f"Number of invalid {column} entries: {len(invalid_entries)}")
    print(invalid_entries[column].value_counts())

def printing_column(df):
    for col in df.columns:
        print(f"{col} unique values:\n{df[col].unique()}\n")

def replacing_invalid(df,name_column, invalid_column):
    mode_value = df[~df[name_column].isin(invalid_column)][name_column].mode().iloc[0]
    df[name_column] = df[name_column].apply(lambda x: mode_value if x in invalid_column else x)
    return df