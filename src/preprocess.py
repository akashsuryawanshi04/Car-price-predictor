"""
preprocess.py
=============
Handles all data loading, cleaning, and feature engineering
for the Quikr Car Price dataset.

Author: Your Name
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")


def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data."""
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def clean_price(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where Price is 'Ask For Price',
    strip commas, and cast to integer.
    """
    df = df[df["Price"] != "Ask For Price"].copy()
    df["Price"] = df["Price"].astype(str).str.replace(",", "", regex=False)
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df.dropna(subset=["Price"], inplace=True)
    print(f"[INFO] After price cleaning: {df.shape[0]} rows")
    return df


def clean_kms_driven(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip ' kms' suffix, remove commas, cast to integer.
    Drop rows where kms_driven is non-numeric.
    """
    df["kms_driven"] = df["kms_driven"].astype(str).str.replace(",", "", regex=False)
    df["kms_driven"] = df["kms_driven"].str.extract(r"(\d+)")
    df["kms_driven"] = pd.to_numeric(df["kms_driven"], errors="coerce")
    df.dropna(subset=["kms_driven"], inplace=True)
    df["kms_driven"] = df["kms_driven"].astype(int)
    print(f"[INFO] After kms_driven cleaning: {df.shape[0]} rows")
    return df


def clean_fuel_type(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing fuel_type."""
    df.dropna(subset=["fuel_type"], inplace=True)
    print(f"[INFO] After fuel_type cleaning: {df.shape[0]} rows")
    return df


def clean_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only 4-digit year values between 1990 and 2024.
    Extract first 4-digit number if needed.
    """
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[(df["year"] >= 1990) & (df["year"] <= 2024)].copy()
    print(f"[INFO] After year cleaning: {df.shape[0]} rows")
    return df


def clean_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Trim car name to first 3 words (model-level granularity).
    Helps reduce cardinality for encoding.
    """
    df["name"] = df["name"].astype(str).apply(
        lambda x: " ".join(x.split()[:3])
    )
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove price outliers using the IQR method.
    Keeps data within [Q1 - 1.5*IQR, Q3 + 1.5*IQR].
    """
    Q1 = df["Price"].quantile(0.25)
    Q3 = df["Price"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = len(df)
    df = df[(df["Price"] >= lower) & (df["Price"] <= upper)].copy()
    print(f"[INFO] Outlier removal: {before - len(df)} rows dropped, {len(df)} remaining")
    return df


def reset_index(df: pd.DataFrame) -> pd.DataFrame:
    df.reset_index(drop=True, inplace=True)
    return df


def full_pipeline(filepath: str) -> pd.DataFrame:
    """
    Run the complete preprocessing pipeline and return a clean DataFrame.
    """
    df = load_data(filepath)

    print("\n── Step 1 : Clean Price ─────────────────────")
    df = clean_price(df)

    print("\n── Step 2 : Clean kms_driven ────────────────")
    df = clean_kms_driven(df)

    print("\n── Step 3 : Clean fuel_type ─────────────────")
    df = clean_fuel_type(df)

    print("\n── Step 4 : Clean year ──────────────────────")
    df = clean_year(df)

    print("\n── Step 5 : Clean name ──────────────────────")
    df = clean_name(df)

    print("\n── Step 6 : Remove Outliers ─────────────────")
    df = remove_outliers(df)

    df = reset_index(df)

    # Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    df = reset_index(df)
    print(f"\n[INFO] Duplicates removed: {before - len(df)}")
    print(f"\n✅ Final clean dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    print(df.dtypes)
    return df


if __name__ == "__main__":
    df = full_pipeline("../data/quikr_car.csv")
    print("\nSample rows:")
    print(df.head())
