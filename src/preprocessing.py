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


def handle_missing_values(df):

    """
    Fill missing values

    Numerical columns -> median
    Categorical columns -> mode
    """
    # Numerical columns
    numerical_cols = df.select_dtypes(
        include=np.number
    ).columns

    for col in numerical_cols:

        df[col] = df[col].fillna(
            df[col].median()
        )

    # Categorical columns
    categorical_cols = df.select_dtypes(
        include='object'
    ).columns

    for col in categorical_cols:

        df[col] = df[col].fillna(
            df[col].mode()[0]
        )

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

def encode_features(df):

    """
    Encode categorical columns
    using Label Encoding
    """

    encoder = LabelEncoder()

    categorical_cols = df.select_dtypes(
        include='object'
    ).columns

    for col in categorical_cols:
        df[col] = encoder.fit_transform(df[col])

    return df

def scale_features(X_train, X_test)
    """
    Scale features using StandardScaler
    """

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_test = scaler.transform(
        X_test
    )

    return X_train, X_test


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into train and test
    """

    X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size=test_size, random_state=random_state   
    )

    return X_train, X_test, y_train, y_test


