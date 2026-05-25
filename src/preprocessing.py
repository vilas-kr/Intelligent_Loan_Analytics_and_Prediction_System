import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file_path):
    """
    Load dataset from CSV file
    """
    df = pd.read_csv(file_path)
    return df


def remove_duplicates(df):
    """
    Remove duplicate rows
    """
    print("Duplicate Rows:", df.duplicated().sum())
    
    df = df.drop_duplicates()
    
    return df


def handle_outliers(df, columns):

    """
    Handle outliers using IQR method
    """

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df[col] = np.where( df[col] > upper, 
                upper, np.where( df[col] < lower, 
                lower, df[col])
            )

    return df




