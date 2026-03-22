"""
eda.py
======
Exploratory Data Analysis for the Quikr Car Price dataset.
Generates plots saved to ../reports/figures/

Author: Your Name
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import full_pipeline

# ── Output folder ────────────────────────────────────────────────────────────
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Plot style ────────────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor":   "#1a1a1a",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#cccccc",
    "ytick.color":      "#cccccc",
    "text.color":       "#ffffff",
    "grid.color":       "#2e2e2e",
    "figure.dpi":       150,
})
ACCENT = "#00c9a7"
WARM   = "#ff6b6b"


def save(fig, name: str):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_price_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f0f0f")

    axes[0].hist(df["Price"], bins=50, color=ACCENT, edgecolor="none", alpha=0.85)
    axes[0].set_title("Price Distribution", fontsize=14, color="#ffffff")
    axes[0].set_xlabel("Price (₹)")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(np.log1p(df["Price"]), bins=50, color=WARM, edgecolor="none", alpha=0.85)
    axes[1].set_title("Log(Price) Distribution", fontsize=14, color="#ffffff")
    axes[1].set_xlabel("log(Price + 1)")
    axes[1].set_ylabel("Frequency")

    fig.suptitle("Car Price — Raw vs Log Scale", fontsize=16, color="#ffffff", y=1.02)
    save(fig, "01_price_distribution.png")


def plot_company_counts(df):
    top = df["company"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#0f0f0f")

    bars = ax.barh(top.index[::-1], top.values[::-1], color=ACCENT, alpha=0.85)
    ax.bar_label(bars, padding=4, color="#ffffff", fontsize=9)
    ax.set_title("Top 15 Car Companies by Listings", fontsize=14, color="#ffffff")
    ax.set_xlabel("Number of Listings")
    save(fig, "02_company_counts.png")


def plot_fuel_type(df):
    counts = df["fuel_type"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#0f0f0f")

    colors = [ACCENT, WARM, "#f9ca24", "#6c5ce7", "#fd79a8"]
    wedges, texts, autotexts = ax.pie(
        counts, labels=counts.index, autopct="%1.1f%%",
        colors=colors[:len(counts)], startangle=140,
        textprops={"color": "#ffffff"}, pctdistance=0.82
    )
    for at in autotexts:
        at.set_fontsize(10)
    ax.set_title("Fuel Type Distribution", fontsize=14, color="#ffffff")
    save(fig, "03_fuel_type_distribution.png")


def plot_price_by_fuel(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f0f0f")

    order = df.groupby("fuel_type")["Price"].median().sort_values(ascending=False).index
    sns.boxplot(data=df, x="fuel_type", y="Price", order=order,
                palette="muted", ax=ax, linewidth=1.2)
    ax.set_title("Price Distribution by Fuel Type", fontsize=14, color="#ffffff")
    ax.set_xlabel("Fuel Type")
    ax.set_ylabel("Price (₹)")
    save(fig, "04_price_by_fuel.png")


def plot_year_trend(df):
    yearly = df.groupby("year")["Price"].median().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("#0f0f0f")

    ax.plot(yearly["year"], yearly["Price"], color=ACCENT,
            lw=2.5, marker="o", markersize=5)
    ax.fill_between(yearly["year"], yearly["Price"], alpha=0.15, color=ACCENT)
    ax.set_title("Median Car Price Over Manufacturing Year", fontsize=14, color="#ffffff")
    ax.set_xlabel("Year")
    ax.set_ylabel("Median Price (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    save(fig, "05_price_by_year.png")


def plot_kms_vs_price(df):
    sample = df.sample(min(500, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f0f0f")

    sc = ax.scatter(sample["kms_driven"], sample["Price"],
                    c=sample["year"], cmap="plasma",
                    alpha=0.7, s=25, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Year", color="#ffffff")
    cbar.ax.yaxis.set_tick_params(color="#ffffff")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#ffffff")

    ax.set_title("KMs Driven vs Price (coloured by Year)", fontsize=14, color="#ffffff")
    ax.set_xlabel("KMs Driven")
    ax.set_ylabel("Price (₹)")
    save(fig, "06_kms_vs_price.png")


def plot_correlation_heatmap(df):
    numeric = df[["year", "Price", "kms_driven"]].copy()
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0f0f0f")

    mask = np.zeros_like(corr, dtype=bool)
    np.fill_diagonal(mask, False)

    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax, linewidths=0.5, linecolor="#2e2e2e",
                annot_kws={"size": 12, "color": "white"})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, color="#ffffff")
    ax.tick_params(axis="x", colors="#cccccc")
    ax.tick_params(axis="y", colors="#cccccc")
    save(fig, "07_correlation_heatmap.png")


def plot_top_companies_price(df):
    top_companies = df["company"].value_counts().head(10).index
    subset = df[df["company"].isin(top_companies)]

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0f0f0f")

    order = subset.groupby("company")["Price"].median().sort_values(ascending=False).index
    sns.boxplot(data=subset, x="company", y="Price", order=order,
                palette="Set2", ax=ax, linewidth=1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_title("Price Distribution — Top 10 Companies", fontsize=14, color="#ffffff")
    ax.set_xlabel("Company")
    ax.set_ylabel("Price (₹)")
    save(fig, "08_top_companies_price.png")


def run_eda(filepath: str = "../data/quikr_car.csv"):
    print("═" * 55)
    print("  EXPLORATORY DATA ANALYSIS — Quikr Car Dataset")
    print("═" * 55)

    df = full_pipeline(filepath)

    print("\n📊 Basic Statistics:")
    print(df.describe().to_string())

    print("\n🎨 Generating plots …")
    plot_price_distribution(df)
    plot_company_counts(df)
    plot_fuel_type(df)
    plot_price_by_fuel(df)
    plot_year_trend(df)
    plot_kms_vs_price(df)
    plot_correlation_heatmap(df)
    plot_top_companies_price(df)

    print(f"\n✅ All plots saved to  {os.path.abspath(FIG_DIR)}")
    return df


if __name__ == "__main__":
    run_eda("../data/quikr_car.csv")
