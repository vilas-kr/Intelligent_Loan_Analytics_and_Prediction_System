import pandas as pd
import numpy as np

def create_high_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create High_Risk target variable.

    High_Risk = 1 if customer has
    previous default history.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """

    df["High_Risk"] = np.where(
        df["Default_History_Count"] > 0,
        1,
        0
    )

    return df


def create_total_applicant_income(df):
    """
    Create a total income feature by combining applicant and coapplicant income.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing 'Applicant_Income' and 'Coapplicant_Income'.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a new column 'Total_Applicant_Income'.
    """
    
    df['Total_Applicant_Income'] = (
        df['Applicant_Income'] +
        df['Coapplicant_Income']
    )
    
    return df 
    
    
def create_emi_to_income(df):
    """
    Create a repayment burden feature by comparing existing EMIs to monthly income.

    The feature is calculated as:
    Existing_EMIs / (Annual_Household_Income / 12)

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing 'Existing_EMIs' and 'Annual_Household_Income'.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a new column 'EMI_to_Income'.
    """
    
    df['EMI_to_Income'] = (
        df['Existing_EMIs'] / (
            df['Annual_Household_Income'].replace(0, np.nan)/12
        )
    )
    
    return df


def create_loan_per_asset(df):
    """
    Create a loan-to-asset ratio feature.

    This feature measures loan amount relative to the applicant's asset value.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing 'Loan_Amount' and 'Asset_Value'.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a new column 'Loan_per_Asset'.
    """
    
    df['Loan_per_Asset'] = (
        df['Loan_Amount']/
        df['Asset_Value'].replace(0, np.nan)
    )
    
    return df
    
    
def create_expense_to_income(df):
    """
    Create an expense burden feature by comparing monthly expenses to monthly income.

    The feature is calculated as:
    Monthly_Expense / (Annual_Household_Income / 12)

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing 'Monthly_Expense' and 'Annual_Household_Income'.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a new column 'Expense_to_Income'.
    """
    
    df['Expense_to_Income'] = df['Monthly_Expense']/(
        df['Annual_Household_Income'].replace(0, np.nan)/12
    )
    
    return df
