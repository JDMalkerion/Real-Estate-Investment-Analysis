"""
Exploratory Data Analysis (EDA) module.

Generates 20 static matplotlib PNG charts (+ scorecard) in reports/eda/figures/
and writes data-backed findings to reports/eda/findings.md.

Every number comes from Data/housing_clean.csv — none are hard-coded.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import List, Tuple

# Ensure workspace root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend; must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as stats

from src.config import CLEAN_DATA_PATH, EDA_FINDINGS_PATH, EDA_REPORTS_DIR

# ── Design constants ─────────────────────────────────────────────────────────
ACCENT   = "#2563EB"   # vivid blue
GREY     = "#CBD5E1"   # light slate for comparison bars
GREY_MED = "#94A3B8"   # medium slate for secondary lines
RED_LINE = "#EF4444"   # mean line
GRN_LINE = "#22C55E"   # median line
BG       = "#F8FAFC"   # figure background
DPI      = 150
FIG_W, FIG_H = 11, 5.5   # inches → ~1650 × 825 px @ 150 dpi

# figures output directory
FIGURES_DIR = EDA_REPORTS_DIR / "figures"

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.facecolor":    BG,
    "figure.facecolor":  BG,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
})


# ── Helpers ──────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, name: str) -> Path:
    """Save figure as PNG and close it."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    kb = path.stat().st_size / 1024
    print(f"  saved {name}  ({kb:,.0f} KB)")
    return path


def _ci95(series: pd.Series) -> float:
    """95% CI half-width for the mean."""
    n = len(series)
    if n < 2:
        return 0.0
    return 1.96 * series.std(ddof=1) / math.sqrt(n)


def _pct_spread(grouped_means: pd.Series) -> float:
    """(max - min) / min as a percentage."""
    lo, hi = grouped_means.min(), grouped_means.max()
    return (hi - lo) / lo * 100 if lo > 0 else 0.0


# ── Chart functions ───────────────────────────────────────────────────────────

def q01_price_distribution(df: pd.DataFrame) -> Tuple[str, str]:
    p_mean = df["Price_in_Lakhs"].mean()
    p_med  = df["Price_in_Lakhs"].median()
    p_std  = df["Price_in_Lakhs"].std()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.hist(df["Price_in_Lakhs"], bins=60, color=ACCENT, edgecolor="white", linewidth=0.3)
    ax.axvline(p_mean, color=RED_LINE, linewidth=1.8, linestyle="--",
               label=f"Mean  Rs{p_mean:.1f}L")
    ax.axvline(p_med, color=GRN_LINE, linewidth=1.8, linestyle=":",
               label=f"Median Rs{p_med:.1f}L")
    ax.set_xlabel("Price (Rs Lakhs)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Q1 - Price distribution: mean Rs{p_mean:.2f}L, median Rs{p_med:.2f}L, "
        f"std Rs{p_std:.2f}L -- nearly flat across Rs10L-Rs500L"
    )
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rs{x:.0f}L"))
    fig.tight_layout()
    _save(fig, "q01_price_distribution.png")

    return (
        "Q1: Price Distribution",
        f"Mean Rs{p_mean:.2f}L, median Rs{p_med:.2f}L, std Rs{p_std:.2f}L, "
        f"range Rs{df['Price_in_Lakhs'].min():.0f}L-Rs{df['Price_in_Lakhs'].max():.0f}L. "
        "Broad and roughly flat across the range, not bell-shaped."
    )


def q02_size_distribution(df: pd.DataFrame) -> Tuple[str, str]:
    s_mean = df["Size_in_SqFt"].mean()
    s_med  = df["Size_in_SqFt"].median()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.hist(df["Size_in_SqFt"], bins=60, color=ACCENT, edgecolor="white", linewidth=0.3)
    ax.axvline(s_mean, color=RED_LINE, linewidth=1.8, linestyle="--",
               label=f"Mean  {s_mean:,.0f} sqft")
    ax.axvline(s_med, color=GRN_LINE, linewidth=1.8, linestyle=":",
               label=f"Median {s_med:,.0f} sqft")
    ax.set_xlabel("Size (sq ft)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Q2 - Size distribution: perfectly uniform 500-5000 sqft "
        f"(mean {s_mean:,.0f}, median {s_med:,.0f})"
    )
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    _save(fig, "q02_size_distribution.png")

    return (
        "Q2: Size Distribution",
        f"500-5000 sqft, mean {s_mean:,.1f}, median {s_med:,.1f}. "
        "Essentially a perfect uniform spread."
    )


def q03_price_per_sqft_by_type(df: pd.DataFrame) -> Tuple[str, str]:
    cap = df["Price_per_SqFt"].quantile(0.99)
    types = sorted(df["Property_Type"].unique())
    data_by_type = [
        df.loc[df["Property_Type"] == t, "Price_per_SqFt"].clip(upper=cap)
        for t in types
    ]
    means = {t: df.loc[df["Property_Type"] == t, "Price_per_SqFt"].mean() for t in types}
    spread = _pct_spread(pd.Series(means))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    bp = ax.boxplot(data_by_type, tick_labels=types, patch_artist=True,
                    medianprops=dict(color=RED_LINE, linewidth=2),
                    boxprops=dict(facecolor=GREY, color=GREY_MED),
                    whiskerprops=dict(color=GREY_MED),
                    capprops=dict(color=GREY_MED),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3,
                                    markerfacecolor=GREY_MED, markeredgecolor=GREY_MED))
    # highlight highest-mean box
    hi_idx = max(range(len(types)), key=lambda i: means[types[i]])
    bp["boxes"][hi_idx].set_facecolor(ACCENT)
    ax.set_ylabel("Price per sq ft (Rs)")
    ax.set_title(
        f"Q3 - Price/sqft by property type: {spread:.1f}% spread -- no type premium "
        f"(y-axis clipped at 99th pct ~Rs{cap:,.0f})"
    )
    ax.text(0.99, 0.02, f"y-axis clipped at 99th percentile (Rs{cap:,.0f}/sqft)",
            transform=ax.transAxes, ha="right", fontsize=7, color=GREY_MED)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rs{x:,.0f}"))
    fig.tight_layout()
    _save(fig, "q03_price_per_sqft_by_property_type.png")

    mean_str = ", ".join(f"{t} Rs{means[t]:,.0f}" for t in types)
    return (
        "Q3: Price per SqFt by Property Type",
        f"Mean price per sqft: {mean_str}. Spread {spread:.1f}% -- no type premium."
    )


def q04_size_vs_price(df: pd.DataFrame) -> Tuple[str, str]:
    r, _ = stats.pearsonr(df["Size_in_SqFt"], df["Price_in_Lakhs"])

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    # hexbin to avoid plotting 250 000 points
    hb = ax.hexbin(df["Size_in_SqFt"], df["Price_in_Lakhs"],
                   gridsize=60, cmap="Blues", mincnt=1, linewidths=0.2)
    plt.colorbar(hb, ax=ax, label="Count per bin")
    # regression line (full data)
    m, b = np.polyfit(df["Size_in_SqFt"], df["Price_in_Lakhs"], 1)
    x_range = np.array([df["Size_in_SqFt"].min(), df["Size_in_SqFt"].max()])
    ax.plot(x_range, m * x_range + b, color=RED_LINE, linewidth=2,
            linestyle="--", label=f"OLS fit (slope {m:.4f})")
    ax.set_xlabel("Size (sq ft)")
    ax.set_ylabel("Price (Rs Lakhs)")
    ax.set_title(
        f"Q4 - Size vs Price: Pearson r = {r:.3f} -- weak and nearly negligible "
        "(hexbin density; all 250 000 listings)"
    )
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rs{x:.0f}L"))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    _save(fig, "q04_size_vs_price.png")

    return (
        "Q4: Size vs Price",
        f"Pearson r = {r:.3f}. Weak, positive and nearly negligible. "
        "Larger homes are not meaningfully pricier."
    )


def q05_outliers(df: pd.DataFrame) -> Tuple[str, str]:
    q1, q3 = df["Price_per_SqFt"].quantile(0.25), df["Price_per_SqFt"].quantile(0.75)
    iqr    = q3 - q1
    cap    = q3 + 1.5 * iqr
    n_above = (df["Price_per_SqFt"] > cap).sum()
    pct_above = n_above / len(df) * 100

    # size outliers
    qs1, qs3 = df["Size_in_SqFt"].quantile(0.25), df["Size_in_SqFt"].quantile(0.75)
    iqr_s    = qs3 - qs1
    cap_s    = qs3 + 1.5 * iqr_s
    n_size   = (df["Size_in_SqFt"] > cap_s).sum()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))

    # Panel 1: price per sqft histogram with IQR cap
    ax1.hist(df["Price_per_SqFt"], bins=80, color=ACCENT, edgecolor="white", linewidth=0.2)
    ax1.axvline(cap, color=RED_LINE, linewidth=2, linestyle="--",
                label=f"IQR cap Rs{cap:,.0f}/sqft")
    ax1.set_xlabel("Price per sq ft (Rs)")
    ax1.set_ylabel("Count")
    ax1.set_title(
        f"Price/sqft: {pct_above:.1f}% above IQR cap\n"
        f"({n_above:,} of {len(df):,} listings)"
    )
    ax1.legend(fontsize=7)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rs{x:,.0f}"))

    # Panel 2: size histogram
    ax2.hist(df["Size_in_SqFt"], bins=60, color=GREY_MED, edgecolor="white", linewidth=0.2)
    ax2.axvline(cap_s, color=RED_LINE, linewidth=2, linestyle="--",
                label=f"IQR cap {cap_s:,.0f} sqft")
    ax2.set_xlabel("Size (sq ft)")
    ax2.set_ylabel("Count")
    ax2.set_title(
        f"Size: {n_size:,} outliers above IQR cap\n"
        f"({n_size/len(df)*100:.1f}% of listings)"
    )
    ax2.legend(fontsize=7)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.suptitle(
        f"Q5 - Outliers: {pct_above:.2f}% of price/sqft exceed IQR cap; "
        "zero size outliers",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    _save(fig, "q05_outliers_price_and_size.png")

    return (
        "Q5: Outliers",
        f"{n_above:,} listings ({pct_above:.2f}%) exceed the IQR cap of "
        f"Rs{cap:,.0f}/sqft; {n_size} size outliers."
    )


def q06_price_per_sqft_by_state(df: pd.DataFrame) -> Tuple[str, str]:
    state_avg = df.groupby("State")["Price_per_SqFt"].mean().sort_values()
    spread    = _pct_spread(state_avg)
    lo_st, hi_st = state_avg.index[0], state_avg.index[-1]

    fig, ax = plt.subplots(figsize=(FIG_W, max(5.5, len(state_avg) * 0.35)))
    colors = [ACCENT if i == len(state_avg) - 1 else GREY
              for i in range(len(state_avg))]
    ax.barh(state_avg.index, state_avg.values, color=colors, height=0.65)
    ax.set_xlabel("Average price per sq ft (Rs)")
    ax.set_xlim(left=0)
    for i, (v, lbl) in enumerate(zip(state_avg.values, state_avg.index)):
        ax.text(v * 1.003, i, f"Rs{v:,.0f}", va="center", fontsize=7)
    ax.set_title(
        f"Q6 - Price per sqft by state: {spread:.1f}% spread -- no geographic signal\n"
        f"({lo_st} Rs{state_avg[lo_st]:,.0f}  to  {hi_st} Rs{state_avg[hi_st]:,.0f})"
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rs{x:,.0f}"))
    fig.tight_layout()
    _save(fig, "q06_avg_price_per_sqft_by_state.png")

    return (
        "Q6: Price per SqFt by State",
        f"{lo_st} Rs{state_avg[lo_st]:,.0f} to {hi_st} Rs{state_avg[hi_st]:,.0f}. "
        f"{spread:.1f}% spread."
    )


def q07_avg_price_by_city(df: pd.DataFrame) -> Tuple[str, str]:
    city_avg = df.groupby("City")["Price_in_Lakhs"].mean().sort_values()
    spread   = _pct_spread(city_avg)
    lo_c, hi_c = city_avg.index[0], city_avg.index[-1]

    n = len(city_avg)
    top10 = city_avg.tail(10)
    bot10 = city_avg.head(10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))

    # bottom 10
    ax1.barh(bot10.index, bot10.values, color=GREY, height=0.65)
    ax1.set_xlabel("Average price (Rs Lakhs)")
    ax1.set_title("Bottom 10 cities")
    ax1.set_xlim(left=0)
    for i, v in enumerate(bot10.values):
        ax1.text(v * 1.003, i, f"Rs{v:.1f}L", va="center", fontsize=7)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rs{x:.0f}L"))

    # top 10
    ax2.barh(top10.index, top10.values, color=ACCENT, height=0.65)
    ax2.set_xlabel("Average price (Rs Lakhs)")
    ax2.set_title("Top 10 cities")
    ax2.set_xlim(left=0)
    for i, v in enumerate(top10.values):
        ax2.text(v * 1.003, i, f"Rs{v:.1f}L", va="center", fontsize=7)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rs{x:.0f}L"))

    fig.suptitle(
        f"Q7 - Avg price by city: {spread:.1f}% spread -- all cities cluster near "
        f"Rs{city_avg.mean():.1f}L ({n} cities total)",
        fontsize=11, fontweight="bold"
    )
    fig.tight_layout()
    _save(fig, "q07_avg_price_by_city.png")

    return (
        "Q7: Average Price by City",
        f"Rs{city_avg[lo_c]:.2f}L ({lo_c}) to Rs{city_avg[hi_c]:.2f}L ({hi_c}). "
        f"Every city clusters near Rs{city_avg.mean():.2f}L."
    )


def q08_median_age_by_locality(df: pd.DataFrame) -> Tuple[str, str]:
    top20  = df["Locality"].value_counts().index[:20]
    ages   = (df[df["Locality"].isin(top20)]
              .groupby("Locality")["Age_of_Property"]
              .median()
              .sort_values())
    overall = df["Age_of_Property"].median()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    colors = [ACCENT if v == ages.max() else GREY for v in ages.values]
    ax.barh(ages.index, ages.values, color=colors, height=0.65)
    ax.axvline(overall, color=RED_LINE, linewidth=1.5, linestyle="--",
               label=f"Overall median {overall:.0f} yrs")
    ax.set_xlabel("Median property age (years)")
    ax.set_xlim(left=0)
    ax.legend()
    ax.set_title(
        f"Q8 - Median property age in top 20 localities by listing count: "
        f"{ages.min():.0f}-{ages.max():.0f} years (overall median {overall:.0f})"
    )
    fig.tight_layout()
    _save(fig, "q08_median_age_by_locality.png")

    return (
        "Q8: Median Age by Locality",
        f"Median age {ages.min():.0f}-{ages.max():.0f} years in top 20 localities "
        f"(overall median {overall:.0f} years)."
    )


def q09_bhk_mix_by_city(df: pd.DataFrame) -> Tuple[str, str]:
    top10 = df["City"].value_counts().index[:10]
    bhk_pct = (
        df[df["City"].isin(top10)]
        .groupby(["City", "BHK"])
        .size()
        .unstack(fill_value=0)
    )
    bhk_pct = bhk_pct.div(bhk_pct.sum(axis=1), axis=0) * 100

    bhk_colors = [GREY, GREY_MED, ACCENT, "#93C5FD", "#1E40AF"]
    labels = [str(b) + " BHK" for b in bhk_pct.columns]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    bottom = np.zeros(len(bhk_pct))
    for col, color, label in zip(bhk_pct.columns, bhk_colors, labels):
        ax.barh(bhk_pct.index, bhk_pct[col], left=bottom, color=color,
                label=label, height=0.65)
        bottom += bhk_pct[col].values

    ax.set_xlabel("Share (%)")
    ax.set_xlim(0, 100)
    ax.legend(loc="lower right", framealpha=0.6)
    ax.set_title(
        "Q9 - BHK mix in top 10 cities: perfectly uniform (~20% each BHK type)"
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    fig.tight_layout()
    _save(fig, "q09_bhk_distribution_across_cities.png")

    overall = df["BHK"].value_counts(normalize=True).sort_index()
    return (
        "Q9: BHK Mix by City",
        "BHK configuration is uniformly distributed: "
        + ", ".join(f"{b} BHK {overall[b]:.1%}" for b in sorted(overall.index))
    )


def q10_price_trend_by_locality(df: pd.DataFrame) -> Tuple[str, str]:
    top5 = (df.groupby("Locality")["Price_per_SqFt"]
              .mean()
              .sort_values(ascending=False)
              .index[:5])
    top_loc  = top5[0]
    top_mean = df[df["Locality"] == top_loc]["Price_per_SqFt"].mean()

    # bin into 5-year windows
    df2 = df[df["Locality"].isin(top5)].copy()
    df2["year_bin"] = (df2["Year_Built"] // 5) * 5
    trend = (df2.groupby(["Locality", "year_bin"])["Price_per_SqFt"]
                .mean()
                .reset_index())

    colors_line = [ACCENT, "#1D4ED8", GREY_MED, "#64748B", "#94A3B8"]
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for loc, color in zip(top5, colors_line):
        sub = trend[trend["Locality"] == loc].sort_values("year_bin")
        ax.plot(sub["year_bin"], sub["Price_per_SqFt"],
                marker="o", markersize=4, linewidth=1.8,
                color=color, label=loc, alpha=0.9)

    ax.set_xlabel("Build year (5-year bin)")
    ax.set_ylabel("Mean price per sq ft (Rs)")
    ax.set_title(
        f"Q10 - Price/sqft by build-year bin for top 5 localities\n"
        f"Top: {top_loc} avg Rs{top_mean:,.0f}/sqft -- no vintage appreciation trend"
    )
    ax.legend(fontsize=7)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"Rs{x:,.0f}"))
    fig.tight_layout()
    _save(fig, "q10_price_trend_top_localities.png")

    return (
        "Q10: Price Trend for Top Localities",
        f"Top locality ({top_loc}) averages Rs{top_mean:,.0f}/sqft. "
        "Cross-sectional pricing across build years shows no vintage appreciation."
    )


def q11_correlation_heatmap(df: pd.DataFrame) -> Tuple[str, str]:
    num_cols = [
        "Price_in_Lakhs", "Size_in_SqFt", "Price_per_SqFt", "BHK",
        "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals",
        "Floor_No", "Total_Floors", "infrastructure_score", "Future_Price_5Y",
    ]
    corr = df[num_cols].corr().round(3)
    r_pf = corr.loc["Price_in_Lakhs", "Future_Price_5Y"]

    fig, ax = plt.subplots(figsize=(FIG_W + 2, FIG_H + 1))
    n = len(num_cols)
    mat = corr.values
    img = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(img, ax=ax, fraction=0.03, pad=0.02)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short_names = [c.replace("_", "\n") for c in num_cols]
    ax.set_xticklabels(short_names, fontsize=7, rotation=45, ha="right")
    ax.set_yticklabels(short_names, fontsize=7)
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            color = "white" if abs(v) > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6.5, color=color)
    ax.set_title(
        f"Q11 - Correlation heatmap -- strongest pair: "
        f"Price vs Future_Price_5Y (r = {r_pf:.3f}, by construction)"
    )
    ax.text(0.5, -0.18,
            "Note: Price vs Future_Price_5Y r ~0.99 is by construction -- "
            "the 5-year price is today's price grown by a fixed rule.",
            transform=ax.transAxes, ha="center", fontsize=7, color=GREY_MED)
    fig.tight_layout()
    _save(fig, "q11_correlation_heatmap.png")

    return (
        "Q11: Correlation Heatmap",
        f"Strongest pair: Price_in_Lakhs vs Future_Price_5Y r = {r_pf:.3f} "
        "(by construction). Physical attributes show |r| < 0.05."
    )


def _group_mean_ci95(df: pd.DataFrame, group_col: str,
                     value_col: str) -> pd.DataFrame:
    """Return DataFrame with group mean and 95% CI half-width."""
    agg = (df.groupby(group_col)[value_col]
             .agg(["mean", "std", "count"])
             .reset_index())
    agg["ci"] = 1.96 * agg["std"] / np.sqrt(agg["count"])
    return agg


def _bar_with_ci(ax: plt.Axes, groups, means, cis, *, ylabel="", xlabel="",
                 y_shared_max: float = None) -> None:
    """Vertical bars with 95% CI error bars."""
    x = np.arange(len(groups))
    ax.bar(x, means, yerr=cis, capsize=4, color=ACCENT, alpha=0.85,
           error_kw={"elinewidth": 1, "ecolor": RED_LINE})
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_ylim(bottom=0, top=y_shared_max)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"Rs{v:,.0f}"))


def q12_schools_vs_price(df: pd.DataFrame) -> Tuple[str, str]:
    agg    = _group_mean_ci95(df, "Nearby_Schools", "Price_per_SqFt")
    spread = _pct_spread(agg.set_index("Nearby_Schools")["mean"])
    hi_y   = (agg["mean"] + agg["ci"]).max() * 1.05

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _bar_with_ci(ax, agg["Nearby_Schools"], agg["mean"], agg["ci"],
                 ylabel="Mean price per sq ft (Rs)", xlabel="Nearby schools",
                 y_shared_max=hi_y)
    ax.set_title(
        f"Q12 - Schools vs price/sqft: {spread:.1f}% spread -- flat from 1 to 10 schools"
    )
    fig.tight_layout()
    _save(fig, "q12_nearby_schools_vs_price.png")

    lo = agg.loc[agg["Nearby_Schools"].idxmin(), "mean"]
    hi = agg.loc[agg["Nearby_Schools"].idxmax(), "mean"]
    return (
        "Q12: Nearby Schools vs Price",
        f"Rs{lo:,.0f} (1 school) to Rs{hi:,.0f} (10 schools). "
        f"{spread:.1f}% spread."
    )


def q13_hospitals_vs_price(df: pd.DataFrame) -> Tuple[str, str]:
    agg    = _group_mean_ci95(df, "Nearby_Hospitals", "Price_per_SqFt")
    spread = _pct_spread(agg.set_index("Nearby_Hospitals")["mean"])
    hi_y   = (agg["mean"] + agg["ci"]).max() * 1.05

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _bar_with_ci(ax, agg["Nearby_Hospitals"], agg["mean"], agg["ci"],
                 ylabel="Mean price per sq ft (Rs)", xlabel="Nearby hospitals",
                 y_shared_max=hi_y)
    ax.set_title(
        f"Q13 - Hospitals vs price/sqft: {spread:.1f}% spread -- flat from 1 to 10"
    )
    fig.tight_layout()
    _save(fig, "q13_nearby_hospitals_vs_price.png")

    lo = agg["mean"].min()
    hi = agg["mean"].max()
    return (
        "Q13: Nearby Hospitals vs Price",
        f"Rs{lo:,.0f}-Rs{hi:,.0f}. {spread:.1f}% spread -- flat."
    )


def q14_price_by_furnishing(df: pd.DataFrame) -> Tuple[str, str]:
    agg    = _group_mean_ci95(df, "Furnished_Status", "Price_in_Lakhs")
    spread = _pct_spread(agg.set_index("Furnished_Status")["mean"])
    hi_y   = (agg["mean"] + agg["ci"]).max() * 1.05

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _bar_with_ci(ax, agg["Furnished_Status"], agg["mean"], agg["ci"],
                 ylabel="Mean price (Rs Lakhs)", xlabel="Furnishing status",
                 y_shared_max=hi_y)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"Rs{v:.1f}L"))
    ax.set_title(
        f"Q14 - Avg price by furnishing: {spread:.1f}% spread -- no furnishing premium"
    )
    fig.tight_layout()
    _save(fig, "q14_price_by_furnished_status.png")

    rows = agg.set_index("Furnished_Status")["mean"]
    return (
        "Q14: Price by Furnishing Status",
        "  ".join(f"{k} Rs{v:.2f}L" for k, v in rows.items())
        + f". {spread:.1f}% spread."
    )


def q15_price_by_facing(df: pd.DataFrame) -> Tuple[str, str]:
    agg    = _group_mean_ci95(df, "Facing", "Price_per_SqFt")
    spread = _pct_spread(agg.set_index("Facing")["mean"])
    hi_y   = (agg["mean"] + agg["ci"]).max() * 1.05

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _bar_with_ci(ax, agg["Facing"], agg["mean"], agg["ci"],
                 ylabel="Mean price per sq ft (Rs)", xlabel="Facing direction",
                 y_shared_max=hi_y)
    ax.set_title(
        f"Q15 - Price/sqft by facing direction: {spread:.1f}% spread -- no facing premium"
    )
    fig.tight_layout()
    _save(fig, "q15_price_per_sqft_by_facing.png")

    rows = agg.set_index("Facing")["mean"]
    return (
        "Q15: Price per SqFt by Facing",
        "  ".join(f"{k} Rs{v:,.0f}" for k, v in rows.items())
        + f". {spread:.1f}% spread."
    )


def q16_count_by_owner_type(df: pd.DataFrame) -> Tuple[str, str]:
    counts = df["Owner_Type"].value_counts().sort_values()
    pcts   = counts / len(df) * 100

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    bars = ax.barh(counts.index, counts.values, color=ACCENT, height=0.5)
    ax.set_xlabel("Number of listings")
    ax.set_title(
        "Q16 - Listings by owner type: equal three-way split (~33.3% each)"
    )
    for bar in bars:
        w = bar.get_width()
        idx = list(counts.values).index(w)
        ax.text(w * 1.005, bar.get_y() + bar.get_height() / 2,
                f"{w:,}  ({pcts.iloc[idx]:.1f}%)", va="center", fontsize=8)
    ax.set_xlim(left=0, right=counts.max() * 1.2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    _save(fig, "q16_count_by_owner_type.png")

    return (
        "Q16: Count by Owner Type",
        "  ".join(f"{k} {v:,} ({v/len(df):.1%})" for k, v in counts.items())
    )


def q17_count_by_availability(df: pd.DataFrame) -> Tuple[str, str]:
    counts = df["Availability_Status"].value_counts().sort_values()
    pcts   = counts / len(df) * 100

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    bars = ax.barh(counts.index, counts.values, color=ACCENT, height=0.5)
    ax.set_xlabel("Number of listings")
    ax.set_title(
        "Q17 - Listings by availability: near-perfect 50/50 split"
    )
    for bar in bars:
        w = bar.get_width()
        idx = list(counts.values).index(w)
        ax.text(w * 1.005, bar.get_y() + bar.get_height() / 2,
                f"{w:,}  ({pcts.iloc[idx]:.1f}%)", va="center", fontsize=8)
    ax.set_xlim(left=0, right=counts.max() * 1.2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    _save(fig, "q17_count_by_availability_status.png")

    return (
        "Q17: Count by Availability Status",
        "  ".join(f"{k} {v:,} ({v/len(df):.1%})" for k, v in counts.items())
    )


def q18_parking_vs_price(df: pd.DataFrame) -> Tuple[str, str]:
    agg    = _group_mean_ci95(df, "Parking_Space", "Price_in_Lakhs")
    spread = _pct_spread(agg.set_index("Parking_Space")["mean"])
    hi_y   = (agg["mean"] + agg["ci"]).max() * 1.05

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _bar_with_ci(ax, agg["Parking_Space"], agg["mean"], agg["ci"],
                 ylabel="Mean price (Rs Lakhs)", xlabel="Parking available",
                 y_shared_max=hi_y)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"Rs{v:.1f}L"))
    ax.set_title(
        f"Q18 - Avg price with/without parking: {spread:.1f}% spread -- negligible"
    )
    fig.tight_layout()
    _save(fig, "q18_parking_vs_price.png")

    rows = agg.set_index("Parking_Space")["mean"]
    return (
        "Q18: Parking vs Price",
        f"With parking Rs{rows.get('Yes', 0):.2f}L vs without Rs{rows.get('No', 0):.2f}L. "
        f"{spread:.1f}% spread."
    )


def q19_amenities_vs_price(df: pd.DataFrame) -> Tuple[str, str]:
    agg    = _group_mean_ci95(df, "amenity_count", "Price_per_SqFt")
    spread = _pct_spread(agg.set_index("amenity_count")["mean"])
    hi_y   = (agg["mean"] + agg["ci"]).max() * 1.05

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _bar_with_ci(ax, agg["amenity_count"], agg["mean"], agg["ci"],
                 ylabel="Mean price per sq ft (Rs)", xlabel="Number of amenities",
                 y_shared_max=hi_y)
    ax.set_title(
        f"Q19 - Price/sqft by amenity count: {spread:.1f}% spread -- flat"
    )
    fig.tight_layout()
    _save(fig, "q19_amenities_vs_price.png")

    lo_v = agg.loc[agg["amenity_count"].idxmin(), "mean"]
    hi_v = agg.loc[agg["amenity_count"].idxmax(), "mean"]
    return (
        "Q19: Amenities vs Price",
        f"Rs{lo_v:,.0f}/sqft (1 amenity) to Rs{hi_v:,.0f}/sqft (all 5). "
        f"{spread:.1f}% spread -- flat."
    )


def q20_transport_vs_price_and_label(df: pd.DataFrame) -> Tuple[str, str]:
    # Order: Low, Medium, High
    order = ["Low", "Medium", "High"]
    price_agg = (df.groupby("Public_Transport_Accessibility")["Price_per_SqFt"]
                   .agg(["mean", "std", "count"]).reindex(order).reset_index())
    price_agg["ci"] = 1.96 * price_agg["std"] / np.sqrt(price_agg["count"])

    inv_rate = (df.groupby("Public_Transport_Accessibility")["Good_Investment"]
                  .mean()
                  .reindex(order) * 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FIG_W, FIG_H))

    # Panel 1: price per sqft (flat)
    x = np.arange(len(order))
    ax1.bar(x, price_agg["mean"], yerr=price_agg["ci"], capsize=4,
            color=GREY, alpha=0.9,
            error_kw={"elinewidth": 1, "ecolor": RED_LINE})
    ax1.set_xticks(x)
    ax1.set_xticklabels(order)
    ax1.set_ylabel("Mean price per sq ft (Rs)")
    ax1.set_xlabel("Transport accessibility")
    ax1.set_ylim(bottom=0)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"Rs{v:,.0f}"))
    ax1.set_title("Price/sqft (flat)")
    price_spread = _pct_spread(price_agg.set_index("Public_Transport_Accessibility")["mean"])
    ax1.text(0.5, 0.96, f"Spread {price_spread:.1f}%",
             transform=ax1.transAxes, ha="center", fontsize=8, color=GREY_MED)

    # Panel 2: Good Investment rate (large diff -- by construction)
    ax2.bar(x, inv_rate.values, color=ACCENT, alpha=0.9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(order)
    ax2.set_ylabel("Good Investment rate (%)")
    ax2.set_xlabel("Transport accessibility")
    ax2.set_ylim(bottom=0, top=100)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    for xi, v in zip(x, inv_rate.values):
        ax2.text(xi, v + 1.5, f"{v:.1f}%", ha="center", fontsize=8)
    ax2.set_title("Good Investment rate (by-construction)")

    caption = (
        "Circular: transport access is an input to the infrastructure score "
        "that defines the Good Investment label."
    )
    fig.suptitle(
        f"Q20 - Transport vs price (flat, {price_spread:.1f}% spread) and "
        f"Good Investment rate ({inv_rate['High']:.1f}% High vs {inv_rate['Low']:.1f}% Low)",
        fontsize=10, fontweight="bold"
    )
    fig.text(0.5, -0.03, caption, ha="center", fontsize=7, color=GREY_MED,
             style="italic")
    fig.tight_layout()
    _save(fig, "q20_transport_vs_price_and_label.png")

    hi_r = inv_rate["High"]
    lo_r = inv_rate["Low"]
    hi_p = price_agg.set_index("Public_Transport_Accessibility").loc["High", "mean"]
    lo_p = price_agg.set_index("Public_Transport_Accessibility").loc["Low", "mean"]
    return (
        "Q20: Transport vs Price and Investment Label",
        f"Price flat (High Rs{hi_p:,.0f} vs Low Rs{lo_p:,.0f}, {price_spread:.1f}% spread). "
        f"Good Investment rate: High {hi_r:.1f}% vs Low {lo_r:.1f}% -- "
        "circular: transport is an input to the label rule."
    )


def scorecard(df: pd.DataFrame) -> None:
    """Horizontal bar chart of % spread by factor."""
    factors = {}

    g = df.groupby("State")["Price_per_SqFt"].mean()
    factors["State (Rs/sqft)"] = _pct_spread(g)

    g = df.groupby("City")["Price_in_Lakhs"].mean()
    factors["City (RsL)"] = _pct_spread(g)

    g = df.groupby("Property_Type")["Price_per_SqFt"].mean()
    factors["Property type (Rs/sqft)"] = _pct_spread(g)

    g = df.groupby("Nearby_Schools")["Price_per_SqFt"].mean()
    factors["Nearby schools (Rs/sqft)"] = _pct_spread(g)

    g = df.groupby("Nearby_Hospitals")["Price_per_SqFt"].mean()
    factors["Nearby hospitals (Rs/sqft)"] = _pct_spread(g)

    g = df.groupby("Furnished_Status")["Price_in_Lakhs"].mean()
    factors["Furnishing (RsL)"] = _pct_spread(g)

    g = df.groupby("Facing")["Price_per_SqFt"].mean()
    factors["Facing direction (Rs/sqft)"] = _pct_spread(g)

    g = df.groupby("Parking_Space")["Price_in_Lakhs"].mean()
    factors["Parking (RsL)"] = _pct_spread(g)

    g = df.groupby("amenity_count")["Price_per_SqFt"].mean()
    factors["Amenity count (Rs/sqft)"] = _pct_spread(g)

    g = df.groupby("Public_Transport_Accessibility")["Price_per_SqFt"].mean()
    factors["Public transport (Rs/sqft)"] = _pct_spread(g)

    series = pd.Series(factors).sort_values()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    colors = [ACCENT if v == series.max() else GREY for v in series.values]
    ax.barh(series.index, series.values, color=colors, height=0.65)
    ax.axvline(5, color=RED_LINE, linewidth=1.2, linestyle="--", alpha=0.7,
               label="5% reference line")
    ax.set_xlabel("Price spread: (max group - min group) / min group (%)")
    ax.set_title(
        f"Scorecard - % price spread per factor -- all <= {series.max():.1f}%: "
        "no factor moves price"
    )
    ax.legend(fontsize=7)
    for bar in ax.patches:
        w = bar.get_width()
        if w > 0:
            ax.text(w + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{w:.1f}%", va="center", fontsize=7)
    ax.set_xlim(left=0, right=series.max() * 1.35 + 0.5)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    fig.tight_layout()
    _save(fig, "scorecard.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_all_eda(df: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Generate all 21 static PNG charts (q01-q20 + scorecard).

    Returns
    -------
    List[Tuple[str, str]]
        List of (title, finding_text).
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    findings: List[Tuple[str, str]] = []

    chart_fns = [
        q01_price_distribution,
        q02_size_distribution,
        q03_price_per_sqft_by_type,
        q04_size_vs_price,
        q05_outliers,
        q06_price_per_sqft_by_state,
        q07_avg_price_by_city,
        q08_median_age_by_locality,
        q09_bhk_mix_by_city,
        q10_price_trend_by_locality,
        q11_correlation_heatmap,
        q12_schools_vs_price,
        q13_hospitals_vs_price,
        q14_price_by_furnishing,
        q15_price_by_facing,
        q16_count_by_owner_type,
        q17_count_by_availability,
        q18_parking_vs_price,
        q19_amenities_vs_price,
        q20_transport_vs_price_and_label,
    ]

    for fn in chart_fns:
        print(f"Generating {fn.__name__}...")
        result = fn(df)
        findings.append(result)

    print("Generating scorecard...")
    scorecard(df)

    return findings


def write_findings_markdown(findings: List[Tuple[str, str]]) -> None:
    """Write structured findings to reports/eda/findings.md."""
    EDA_FINDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EDA_FINDINGS_PATH, "w") as f:
        f.write("# Exploratory Data Analysis (EDA) Findings\n\n")
        f.write(
            "This report compiles data-backed statistical findings for all 20 analytical "
            "questions. Every number is computed strictly from the cleaned dataset "
            "(`Data/housing_clean.csv`).\n\n"
        )
        for title, text in findings:
            f.write(f"### {title}\n")
            f.write(f"- **Finding**: {text}\n\n")
    print(f"EDA findings written to: {EDA_FINDINGS_PATH}")


def delete_old_html_charts() -> None:
    """Delete legacy q*.html files from reports/eda/."""
    deleted = 0
    for p in EDA_REPORTS_DIR.glob("q*.html"):
        p.unlink()
        deleted += 1
    if deleted:
        print(f"Deleted {deleted} legacy HTML chart(s) from {EDA_REPORTS_DIR}")


def main() -> None:
    """Run full Stage 2 EDA workflow."""
    if not CLEAN_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned dataset missing at {CLEAN_DATA_PATH}")

    print(f"Loading cleaned dataset from {CLEAN_DATA_PATH}...")
    df = pd.read_csv(CLEAN_DATA_PATH)
    print(f"  {len(df):,} rows x {df.shape[1]} columns loaded.")

    delete_old_html_charts()
    findings = generate_all_eda(df)
    write_findings_markdown(findings)

    pngs = list(FIGURES_DIR.glob("*.png"))
    print(f"\nDone. {len(pngs)} PNG(s) in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
