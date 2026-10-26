"""Module 4: decision-support KPIs, risk/review queue, interpretations."""
from __future__ import annotations

import pandas as pd
import plotly.express as px


def kpis(scored: pd.DataFrame) -> dict:
    n = len(scored)
    high = int((scored["risk_band"] == "High").sum())
    med = int((scored["risk_band"] == "Medium").sum())
    return {
        "Reservations scored": n,
        "High-risk (review)": high,
        "Medium-risk": med,
        "Avg predicted risk": f"{scored['risk_probability'].mean():.1%}",
        "Booked hours": int(scored["duration_hours"].sum()),
        "Hours at high risk": int(scored.loc[scored["risk_band"] == "High", "duration_hours"].sum()),
    }


def review_queue(scored: pd.DataFrame, top: int = 50) -> pd.DataFrame:
    cols = [c for c in ["reservation_id", "facility_type", "building", "day_of_week",
                        "start_hour", "duration_hours", "requester_type",
                        "risk_probability", "risk_band", "recommended_follow_up"]
            if c in scored.columns]
    q = scored[scored["risk_band"].isin(["High", "Medium"])]
    return q.sort_values("risk_probability", ascending=False)[cols].head(top)


def facility_risk_fig(scored: pd.DataFrame):
    g = (scored.groupby("facility_type")
               .agg(avg_risk=("risk_probability", "mean"),
                    high_risk=("risk_band", lambda s: (s == "High").sum()),
                    hours=("duration_hours", "sum")).reset_index())
    fig = px.bar(g, x="facility_type", y="avg_risk", hover_data=["high_risk", "hours"],
                 title="Average predicted risk by facility type")
    fig.update_yaxes(tickformat=".0%")
    return fig


def band_fig(scored: pd.DataFrame):
    order = ["Low", "Medium", "High"]
    c = scored["risk_band"].value_counts().reindex(order, fill_value=0).reset_index()
    c.columns = ["band", "count"]
    return px.bar(c, x="band", y="count", color="band",
                  color_discrete_map={"Low": "#2e7d32", "Medium": "#f9a825", "High": "#c62828"},
                  title="Reservations by risk band")


def utilization_fig(scored: pd.DataFrame):
    g = scored.groupby(["day_of_week", "risk_band"])["duration_hours"].sum().reset_index()
    return px.bar(g, x="day_of_week", y="duration_hours", color="risk_band",
                  category_orders={"day_of_week": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
                                   "risk_band": ["Low", "Medium", "High"]},
                  title="Booked hours by day and risk band")


def interpretations(scored: pd.DataFrame) -> list[str]:
    msgs = []
    top = scored.groupby("facility_type")["risk_probability"].mean().sort_values(ascending=False)
    msgs.append(f"{top.index[0]} has the highest average predicted risk ({top.iloc[0]:.0%}); "
                "prioritise reconfirmation there.")
    hi = scored[scored["risk_band"] == "High"]
    if len(hi):
        share = hi["duration_hours"].sum() / scored["duration_hours"].sum()
        msgs.append(f"High-risk bookings hold {share:.0%} of booked hours: candidates for waitlist "
                    "backups, never for automatic cancellation.")
        day = hi["day_of_week"].value_counts().idxmax()
        msgs.append(f"Most high-risk bookings fall on {day}; consider a reminder sweep the day before.")
    else:
        msgs.append("No reservations exceed the high-risk threshold.")
    return msgs
