"""Campus Facility Reservation Monitoring - Streamlit deployment of the midterm model."""
from datetime import datetime

import pandas as pd
import streamlit as st

from modules import dashboard, exploratory_analysis as eda, prediction, system_design
from modules.config import (CATEGORY_VALUES, DIAGRAM_PATH, FEATURES, NUMERIC_BOUNDS,
                            RISK_THRESHOLDS, SAMPLE_DATA_PATH, TARGET)
from modules.data_processing import (clean_for_inference, load_csv, quality_summary,
                                     validate_schema)

st.set_page_config(page_title="Facility Reservation Risk System", layout="wide")

PAGES = ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"]
page = st.sidebar.selectbox("Module", PAGES)


@st.cache_resource
def get_model():
    return prediction.load_pipeline()


@st.cache_data
def get_metadata():
    return prediction.load_metadata()


@st.cache_data
def get_sample():
    return pd.read_csv(SAMPLE_DATA_PATH)


@st.cache_data
def score(df: pd.DataFrame) -> pd.DataFrame:
    return prediction.predict_batch(get_model(), df)


def current_df():
    """Dataset chosen on the Dataset page (uploaded or sample)."""
    return st.session_state.get("df")


def need_data():
    st.warning("No dataset loaded yet. Go to **Dataset** and upload a CSV or load the synthetic sample.")
    st.stop()


meta = get_metadata()
st.sidebar.caption(f"Model: {meta['model_version']}  \nData: {meta['data_source']}")
st.sidebar.info("Predictions are decision support. A facility administrator makes every final decision.")

# ---------------------------------------------------------------- Home
if page == "Home":
    st.title("Campus Facility Reservation Monitoring")
    st.write("Predicts which reservations are likely to be **cancelled or become no-shows**, "
             "so facility administrators can follow up early and reduce wasted capacity.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Model version", meta["model_version"])
    c2.metric("ROC-AUC (test)", f"{meta['metrics']['roc_auc']:.2f}")
    c3.metric("Test rows", meta["metrics"]["test_rows"])
    st.markdown("""
**How to use**
1. **Dataset**: upload a reservation CSV (or load the sample) and review validation.
2. **Data Analysis**: explore distributions and interactive charts.
3. **Prediction**: score one reservation and see model metrics.
4. **Dashboard**: review the risk queue and record the final human decision.
5. **System Design**: requirements, actors, use case diagram, workflow.
""")
    st.caption("Differences from the midterm: see README (synthetic stand-in data/model until replaced).")

# ---------------------------------------------------------------- Dataset
elif page == "Dataset":
    st.header("Dataset Upload and Validation")
    st.write("Required columns: " + ", ".join(f"`{c}`" for c in FEATURES) +
             f". Optional: `{TARGET}` (for EDA), `reservation_id`.")
    up = st.file_uploader("Upload CSV", type="csv")
    if st.button("Load synthetic sample instead"):
        st.session_state["df"] = get_sample()
        st.session_state["source"] = "synthetic sample"
    if up is not None:
        df, err = load_csv(up)
        if err:
            st.error(err)
        else:
            res = validate_schema(df)
            if res["errors"]:
                for e in res["errors"]:
                    st.error(e)
                st.session_state.pop("df", None)
            else:
                st.session_state["df"] = df.drop(columns=res["leakage_found"])
                st.session_state["source"] = up.name
                st.success(f"Loaded {up.name}")
    df = current_df()
    if df is not None:
        st.caption(f"Source: {st.session_state.get('source')}")
        res = validate_schema(df)
        for w in res["warnings"]:
            st.warning(w)
        q = quality_summary(df)
        a, b, c, d = st.columns(4)
        a.metric("Rows", q["rows"]); b.metric("Columns", q["columns"])
        c.metric("Duplicate rows", q["duplicate_rows"]); d.metric("Missing cells", q["total_nulls"])
        if q["total_nulls"]:
            st.dataframe(q["null_table"])
        else:
            st.success("No missing values.")
        st.dataframe(df.head(20), use_container_width=True)

# ---------------------------------------------------------------- EDA
elif page == "Data Analysis":
    st.header("Exploratory Data Analysis")
    df = current_df()
    if df is None:
        need_data()
    if TARGET not in df.columns:
        st.error(f"EDA needs the target column '{TARGET}'.")
        st.stop()
    st.subheader("Descriptive statistics")
    st.dataframe(eda.describe(df), use_container_width=True)
    left, right = st.columns(2)
    left.plotly_chart(eda.class_distribution_fig(df), use_container_width=True)
    cat = right.selectbox("Category", eda.categorical_options())
    right.plotly_chart(eda.rate_by_category_fig(df, cat), use_container_width=True)
    num = st.selectbox("Numeric feature", list(NUMERIC_BOUNDS))
    st.plotly_chart(eda.numeric_distribution_fig(df, num), use_container_width=True)
    l2, r2 = st.columns(2)
    l2.plotly_chart(eda.correlation_fig(df), use_container_width=True)
    r2.plotly_chart(eda.hour_day_heatmap_fig(df), use_container_width=True)

# ---------------------------------------------------------------- Prediction
elif page == "Prediction":
    st.header("Predictive Modeling")
    model = get_model()
    tab1, tab2 = st.tabs(["Predict a reservation", "Model metrics"])
    with tab1:
        with st.form("predict"):
            c = st.columns(3)
            vals = {}
            for i, col in enumerate(["facility_type", "building", "day_of_week",
                                     "requester_type", "semester_period"]):
                vals[col] = c[i % 3].selectbox(col, CATEGORY_VALUES[col])
            for i, (col, (lo, hi)) in enumerate(NUMERIC_BOUNDS.items()):
                vals[col] = c[i % 3].number_input(col, value=float(lo), step=1.0)
            go = st.form_submit_button("Predict")
        if go:
            problems = prediction.validate_inputs(vals)
            if problems:
                for p in problems:
                    st.error(p)
            else:
                out = prediction.predict_one(model, vals)
                st.subheader("Result")
                st.write("**Predicted class:** " +
                         ("Cancelled / No-show" if out["label"] else "Honoured"))
                if "probability" in out:
                    st.progress(out["probability"], text=f"Risk probability {out['probability']:.1%}")
                    st.write(f"**Risk band:** {out['band']}  \n**Recommended human follow-up:** {out['follow_up']}")
                st.caption(f"Model {meta['model_version']} · predicted {out['timestamp']} · "
                           "requires facility administrator review before any action.")
                st.session_state.setdefault("samples", []).append({**vals, **out})
        if st.session_state.get("samples"):
            st.subheader("Predictions this session")
            st.dataframe(pd.DataFrame(st.session_state["samples"]), use_container_width=True)
        with st.expander("Limitations"):
            for l in prediction.LIMITATIONS:
                st.write("- " + l)
    with tab2:
        m = meta["metrics"]
        cols = st.columns(5)
        for col, k in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
            col.metric(k, f"{m[k]:.3f}")
        cm = pd.DataFrame(m["confusion_matrix"], index=["Actual honoured", "Actual cancel/no-show"],
                          columns=["Pred honoured", "Pred cancel/no-show"])
        st.dataframe(cm)
        st.caption(f"{meta['model_type']} · trained {meta['trained_at']} · target `{meta['target']}`")

# ---------------------------------------------------------------- Dashboard
elif page == "Dashboard":
    st.header("Decision-Support Dashboard")
    df = current_df()
    if df is None:
        need_data()
    inputs = clean_for_inference(df)
    if "reservation_id" in df.columns:
        inputs.insert(0, "reservation_id", df["reservation_id"])
    scored = score(inputs)
    k = dashboard.kpis(scored)
    for col, (name, val) in zip(st.columns(len(k)), k.items()):
        col.metric(name, val)
    l, r = st.columns(2)
    l.plotly_chart(dashboard.band_fig(scored), use_container_width=True)
    r.plotly_chart(dashboard.facility_risk_fig(scored), use_container_width=True)
    st.plotly_chart(dashboard.utilization_fig(scored), use_container_width=True)
    st.subheader("Interpretation")
    for msg in dashboard.interpretations(scored):
        st.write("- " + msg)
    st.subheader("Risk / review queue")
    st.caption(f"High ≥ {RISK_THRESHOLDS['high']:.0%}, Medium ≥ {RISK_THRESHOLDS['medium']:.0%}. "
               "Recommendations are suggestions for a human reviewer.")
    queue = dashboard.review_queue(scored)
    st.dataframe(queue, use_container_width=True)

    st.subheader("Record final decision (Facility Administrator)")
    if "reservation_id" in queue.columns and len(queue):
        with st.form("decision"):
            who = st.text_input("Administrator name")
            rid = st.selectbox("Reservation", queue["reservation_id"])
            dec = st.selectbox("Decision", system_design.DECISIONS)
            note = st.text_input("Note (optional)")
            ok = st.form_submit_button("Save decision")
        if ok:
            if not who.strip():
                st.error("Administrator name is required to log a decision.")
            else:
                st.session_state.setdefault("decisions", []).append({
                    "reservation_id": rid, "decision": dec, "note": note, "decided_by": who.strip(),
                    "decided_at": datetime.now().isoformat(timespec="seconds"),
                    "model_version": meta["model_version"]})
                st.success("Decision logged.")
        log = st.session_state.get("decisions")
        if log:
            logdf = pd.DataFrame(log)
            st.dataframe(logdf, use_container_width=True)
            st.download_button("Download decision log (CSV)", logdf.to_csv(index=False),
                               "decision_log.csv", "text/csv")
    else:
        st.info("Add a `reservation_id` column to enable decision logging.")
    st.download_button("Download scored data (CSV)", scored.to_csv(index=False),
                       "scored_reservations.csv", "text/csv")

# ---------------------------------------------------------------- System Design
else:
    st.header("System Analysis and Design Integration")
    st.subheader("Functional requirements")
    st.table(pd.DataFrame(system_design.FUNCTIONAL, columns=["ID", "Requirement", "Implemented in"]))
    st.subheader("Non-functional requirements")
    st.table(pd.DataFrame(system_design.NON_FUNCTIONAL, columns=["ID", "Category", "Requirement"]))
    st.subheader("Actors and access")
    st.table(pd.DataFrame(system_design.ACTOR_ACCESS,
                          columns=["Actor", "Main use case", "Can view results", "Records final decision"]))
    st.subheader("Updated use case diagram")
    if DIAGRAM_PATH.exists():
        st.image(str(DIAGRAM_PATH))
    else:
        st.error("assets/use_case_diagram.png is missing. Run make_diagram.py.")
    st.subheader("Analytical process / decision workflow")
    for s in system_design.WORKFLOW_STEPS:
        st.write(s)
