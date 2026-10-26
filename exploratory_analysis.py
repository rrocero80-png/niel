"""Module 2: descriptive statistics, class distribution, interactive Plotly charts."""
from __future__ import annotations

import pandas as pd
import plotly.express as px

from .config import CATEGORICAL, NUMERIC, TARGET

LABELS = {0: "Honoured", 1: "Cancelled / No-show"}


def describe(df: pd.DataFrame) -> pd.DataFrame:
    return df.describe(include="all").T


def class_distribution_fig(df: pd.DataFrame):
    counts = df[TARGET].map(LABELS).value_counts().reset_index()
    counts.columns = ["outcome", "count"]
    return px.pie(counts, names="outcome", values="count", hole=0.45,
                  title="Class distribution (target)")


def rate_by_category_fig(df: pd.DataFrame, col: str):
    g = (df.groupby(col)[TARGET].agg(["mean", "count"]).reset_index()
           .rename(columns={"mean": "cancel_rate"}))
    fig = px.bar(g, x=col, y="cancel_rate", hover_data=["count"],
                 title=f"Cancellation / no-show rate by {col}")
    fig.update_yaxes(tickformat=".0%", title="rate")
    return fig


def numeric_distribution_fig(df: pd.DataFrame, col: str):
    d = df.assign(outcome=df[TARGET].map(LABELS))
    return px.histogram(d, x=col, color="outcome", barmode="overlay", opacity=0.65,
                        nbins=30, title=f"Distribution of {col} by outcome")


def correlation_fig(df: pd.DataFrame):
    cols = [c for c in NUMERIC + [TARGET] if c in df.columns]
    return px.imshow(df[cols].corr().round(2), text_auto=True, aspect="auto",
                     color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                     title="Correlation of numeric features with outcome")


def hour_day_heatmap_fig(df: pd.DataFrame):
    pivot = df.pivot_table(index="day_of_week", columns="start_hour",
                           values=TARGET, aggfunc="mean")
    order = [d for d in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"] if d in pivot.index]
    return px.imshow(pivot.loc[order], aspect="auto", color_continuous_scale="Reds",
                     title="Cancellation rate: day vs start hour",
                     labels={"color": "rate"})


def categorical_options() -> list[str]:
    return CATEGORICAL
