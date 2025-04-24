import math
import matplotlib.pyplot as plt
import seaborn as sns

## This py file contains all functions which were used in the jupyter notebook BasicVisualization.ipynb

## This function gets the plot of the columns
def get_plots(df):
    # Separate columns by datatype
    object_columns = df.select_dtypes(include='object').columns
    numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

    def plot_group(columns, plot_func, title):
        if len(columns) == 0:
            return

        n_cols = 4
        n_rows = math.ceil(len(columns) / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axes = axes.flatten()

        for ax, col in zip(axes, columns):
            plot_func(ax, col)
        
        for i in range(len(columns), len(axes)):
            fig.delaxes(axes[i])

        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        plt.show()

    # Define plotting functions
    def plot_object(ax, col):
        sns.countplot(data=df, x=col, order=df[col].value_counts().index, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.tick_params(axis='x', rotation=45)

    def plot_numeric(ax, col):
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(f"Histogram of {col}")

    plot_group(object_columns, plot_object, "Categorical Columns")
    plot_group(numeric_columns, plot_numeric, "Numeric Columns")

## function which Count values
def val_count(df, column_name):
    counts = df[column_name].value_counts()
    print(counts)

## drop values by providing values, column and a dataframe
def drop_rows_by_values(df, column, values):
    
    ##Drops rows from the DataFrame where the specified column has one or more of the given values.    
    ##Returns:
        ##pd.DataFrame: A new DataFrame with the specified rows removed.
    
    # Ensure values is a list, even if a single value is passed
    if not isinstance(values, (list, set, tuple)):
        values = [values]
    
    # Drop the rows
    return df[~df[column].isin(values)]

## Function to swap columns
def swap_columns(df, col1, col2):
    ##Swaps the positions of two columns in a DataFrame.
    ##Returns:
        ##pd.DataFrame: A new DataFrame with the columns swapped.
    
    cols = list(df.columns)
    idx1, idx2 = cols.index(col1), cols.index(col2)
    cols[idx1], cols[idx2] = cols[idx2], cols[idx1]
    return df[cols]

## function that label encodes the columns
def encode_labeling(df, column, mapping):
    ##Label encodes the specified column using a provided mapping dictionary.
    ##Returns:
        ##pd.DataFrame: The DataFrame with the encoded column.
    df[column + '_encoded'] = df[column].map(mapping)
    return df

## function that plot all features against target column
def plot_against_target(df, target_col, plots_per_row=3):
    # Drop rows with missing target values
    df = df.dropna(subset=[target_col])
    
    # Separate columns by numeric and object type
    numeric_cols = df.select_dtypes(include='number').columns.drop(target_col, errors='ignore')
    object_cols = df.select_dtypes(include='object').columns

    ## For numeric type column
    total_numeric = len(numeric_cols)
    if total_numeric > 0:
        rows = math.ceil(total_numeric / plots_per_row)
        plt.figure(figsize=(plots_per_row * 5, rows * 4))
        for i, col in enumerate(numeric_cols):
            plt.subplot(rows, plots_per_row, i + 1)
            ##sns.scatterplot(data=df, x=col, y=target_col)
            ##sns.histplot(data=df, x=col, kde=True)
            sns.histplot(data=df, x=col, hue=target_col, bins=30, kde=True, stat="density", common_norm=False)
            plt.title(f"{col} vs {target_col}")
            plt.grid(True)
        plt.tight_layout()
        plt.show()

    ## For object type column
    total_object = len(object_cols)
    if total_object > 0:
        rows = math.ceil(total_object / plots_per_row)
        plt.figure(figsize=(plots_per_row * 5, rows * 5))
        for i, col in enumerate(object_cols):
            plt.subplot(rows, plots_per_row, i + 1)
            counts = df.groupby([col, target_col]).size().unstack(fill_value=0)
            counts.plot(kind='bar', ax=plt.gca())
            plt.title(f"{col} grouped by {target_col}")
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
