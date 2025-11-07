import streamlit as st
import pandas as pd
import numpy as np
import io
from typing import Dict, Tuple, List
from dataclasses import dataclass

# Viz
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# ML
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ---- Page config ----
st.set_page_config(page_title="Universal Bank – Personal Loan Propensity (DT/RF/GBT)",
                   page_icon="💳", layout="wide")

st.title("💳 Universal Bank — Personal Loan Propensity Dashboard")
st.caption("Head of Marketing toolkit: insights → models → deployment")

# --------- Utilities ---------
DATA_COLUMNS = [
    "ID","Personal Loan","Age","Experience","Income","Zip code","Family","CCAvg",
    "Education","Mortgage","Securities","CDAccount","Online","CreditCard"
]

REQUIRED_FEATURES = [
    "Age","Experience","Income","Family","CCAvg","Education","Mortgage",
    "Securities","CDAccount","Online","CreditCard"
]

TARGET_COL = "Personal Loan"
ID_COL = "ID"
ZIP_COL = "Zip code"
RANDOM_STATE = 42

@dataclass
class ModelArtifacts:
    models: Dict[str, Pipeline]
    metrics_df: pd.DataFrame
    results: Dict[str, dict]
    feature_names: List[str]

@st.cache_data
def _clean_universal_bank(df: pd.DataFrame) -> pd.DataFrame:
    # Check columns
    missing = [c for c in DATA_COLUMNS if c not in df.columns]
    if missing:
        # Try to be forgiving on case/spacing by renaming similar columns
        df = df.copy()
        rename_map = {}
        for c in df.columns:
            cl = c.strip().lower().replace(" ", "")
            for need in DATA_COLUMNS:
                nl = need.strip().lower().replace(" ", "")
                if cl == nl:
                    rename_map[c] = need
        if rename_map:
            df = df.rename(columns=rename_map)
        missing = [c for c in DATA_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    # Coerce numerics
    numeric_cols = ["Age","Experience","Income","Family","CCAvg","Education","Mortgage",
                    "Securities","CDAccount","Online","CreditCard"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fix negative Experience
    if (df["Experience"] < 0).any():
        med = df.loc[df["Experience"]>=0, "Experience"].median()
        df.loc[df["Experience"]<0, "Experience"] = med

    # Drop rows with missing target (if present)
    if TARGET_COL in df.columns:
        df = df.dropna(subset=[TARGET_COL])

    # Fill remaining numeric NaNs
    for c in numeric_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    return df

@st.cache_data
def _synthesize_sample(n=5000, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(23, 65, n)
    exp = np.clip(age - rng.integers(18, 30, n), 0, None)
    income = rng.normal(100, 45, n).clip(10, 400)   # $000
    family = rng.integers(1, 5, n)
    ccavg = np.abs(rng.normal(2.0, 1.5, n)).clip(0.1, 20)  # $000
    education = rng.choice([1,2,3], size=n, p=[0.45, 0.35, 0.20])
    mort = np.abs(rng.normal(80, 120, n)).clip(0, 700)  # $000
    securities = rng.integers(0, 2, n)
    cd = rng.integers(0, 2, n)
    online = rng.integers(0, 2, n)
    creditcard = rng.integers(0, 2, n)
    # Propensity function (simulate realistic signals)
    logit = (-3.0
             + 0.012*income
             + 0.25*cd
             + 0.15*securities
             + 0.18*online
             + 0.09*creditcard
             + 0.08*ccavg
             + 0.04*(education==3)
             - 0.01*family
             + 0.005*mort)
    p = 1/(1+np.exp(-logit))
    y = (rng.uniform(0,1,n) < p).astype(int)

    df = pd.DataFrame({
        "ID": np.arange(1, n+1),
        "Personal Loan": y,
        "Age": age,
        "Experience": exp,
        "Income": income.round(2),
        "Zip code": rng.integers(10000, 99999, n),
        "Family": family,
        "CCAvg": ccavg.round(2),
        "Education": education,
        "Mortgage": mort.round(0),
        "Securities": securities,
        "CDAccount": cd,
        "Online": online,
        "CreditCard": creditcard
    })
    return df

@st.cache_resource
def _build_pipelines():
    identity = FunctionTransformer(lambda X: X)
    dt = Pipeline([("identity", identity), ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))])
    rf = Pipeline([("identity", identity), ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=300))])
    gbt = Pipeline([("identity", identity), ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))])

    dt_grid = {
        "clf__criterion": ["gini", "entropy"],
        "clf__max_depth": [None, 4, 6, 8, 10],
        "clf__min_samples_split": [2, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 5, 10],
    }
    rf_grid = {
        "clf__max_depth": [None, 6, 10, 14],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 5],
        "clf__max_features": ["sqrt", "log2", None],
    }
    gbt_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__learning_rate": [0.05, 0.1, 0.2],
        "clf__max_depth": [2, 3, 4],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 5],
    }
    return {
        "Decision Tree": (dt, dt_grid),
        "Random Forest": (rf, rf_grid),
        "Gradient Boosted Tree": (gbt, gbt_grid)
    }

def _fit_with_cv(name, pipe, param_grid, X_train, y_train, cv=5):
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(estimator=pipe, param_grid=param_grid,
                      scoring="roc_auc", cv=cv_obj, n_jobs=-1, refit=True, verbose=0)
    gs.fit(X_train, y_train)
    return gs.best_estimator_, gs.best_score_, gs.best_params_

def _evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    yhat_tr = model.predict(X_tr)
    yhat_te = model.predict(X_te)
    proba_tr = model.predict_proba(X_tr)[:, 1]
    proba_te = model.predict_proba(X_te)[:, 1]

    acc_tr = accuracy_score(y_tr, yhat_tr)
    acc_te = accuracy_score(y_te, yhat_te)
    prec = precision_score(y_te, yhat_te, zero_division=0)
    rec = recall_score(y_te, yhat_te, zero_division=0)
    f1 = f1_score(y_te, yhat_te, zero_division=0)
    auc_te = roc_auc_score(y_te, proba_te)

    fpr_tr, tpr_tr, _ = roc_curve(y_tr, proba_tr)
    fpr_te, tpr_te, _ = roc_curve(y_te, proba_te)

    cm_tr = confusion_matrix(y_tr, yhat_tr)
    cm_te = confusion_matrix(y_te, yhat_te)

    return {
        "name": name,
        "acc_train": acc_tr, "acc_test": acc_te,
        "precision": prec, "recall": rec, "f1": f1, "auc_test": auc_te,
        "fpr_train": fpr_tr, "tpr_train": tpr_tr, "fpr_test": fpr_te, "tpr_test": tpr_te,
        "cm_train": cm_tr, "cm_test": cm_te,
        "model": model
    }

def _roc_plotly(results: Dict[str, dict]) -> go.Figure:
    fig = go.Figure()
    palette = {
        "Decision Tree":"blue",
        "Random Forest":"orange",
        "Gradient Boosted Tree":"green"
    }
    for k, r in results.items():
        fig.add_trace(go.Scatter(x=r["fpr_test"], y=r["tpr_test"],
                                 mode="lines",
                                 name=f"{k} (AUC={r['auc_test']:.3f})",
                                 line=dict(color=palette.get(k,None))))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line=dict(dash="dash")))
    fig.update_layout(title="ROC Curves (Test)", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return fig

def _cm_plotly(cm: np.ndarray, title: str) -> go.Figure:
    z = cm.astype(int)
    fig = go.Figure(data=go.Heatmap(z=z, x=[0,1], y=[0,1], text=z, texttemplate="%{text}", colorscale="Blues"))
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="True")
    return fig

def _feature_importance_plotly(model: Pipeline, feature_names: List[str], title: str) -> go.Figure:
    importances = model.named_steps["clf"].feature_importances_
    order = np.argsort(importances)[::-1]
    fig = go.Figure(go.Bar(x=[feature_names[i] for i in order], y=importances[order]))
    fig.update_layout(title=title, xaxis_title="Feature", yaxis_title="Importance")
    return fig

def _lift_gain_figure(y_true: np.ndarray, y_score: np.ndarray) -> go.Figure:
    # Compute decile lift and cumulative gains
    df = pd.DataFrame({"y": y_true, "p": y_score}).sort_values("p", ascending=False).reset_index(drop=True)
    df["decile"] = (np.floor(np.arange(len(df)) / (len(df)/10))).astype(int).clip(0,9)
    dec = df.groupby("decile").agg(positives=("y","sum"), total=("y","size")).reset_index()
    dec["cum_pos"] = dec["positives"].cumsum()
    total_pos = dec["positives"].sum()
    dec["cum_perc_customers"] = ((dec["total"].cumsum())/len(df))*100
    dec["cum_perc_positives"] = (dec["cum_pos"]/total_pos)*100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dec["cum_perc_customers"], y=dec["cum_perc_positives"],
                             mode="lines+markers", name="Cumulative Gains"))
    fig.add_trace(go.Scatter(x=[0,100], y=[0,100], mode="lines", name="Baseline", line=dict(dash="dash")))
    fig.update_layout(title="Cumulative Gains (Deciles)", xaxis_title="% of Customers (sorted by score)",
                      yaxis_title="% of Responders Captured")
    return fig

def _bin_series(x, bins, labels=None):
    out = pd.cut(x, bins=bins, labels=labels, include_lowest=True, duplicates='drop')
    return out

def _download_df_button(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

# --------- Sidebar: data source ---------
st.sidebar.header("📥 Data")
uploaded = st.sidebar.file_uploader("Upload Universal Bank CSV", type=["csv"])
if uploaded:
    df_raw = pd.read_csv(uploaded)
    st.sidebar.success("File loaded.")
else:
    st.sidebar.info("No file uploaded — using a synthetic sample matching the schema.")
    df_raw = _synthesize_sample(n=5000)

# Clean
try:
    df = _clean_universal_bank(df_raw)
except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

# Basic preview
with st.expander("🔎 Data preview & dictionary", expanded=False):
    st.write(df.head())
    dict_df = pd.DataFrame({
        "Field": ["ID","Personal Loan","Age","Experience","Income","Zip code","Family","CCAvg","Education","Mortgage","Securities","CDAccount","Online","CreditCard"],
        "Description": [
            "Unique identifier",
            "Accepted personal loan? (1 Yes, 0 No)",
            "Customer age",
            "Years of professional experience",
            "Annual income in $000",
            "Home address zip code",
            "Family size",
            "Avg credit card spend per month in $000",
            "Education level (1=UG,2=Grad,3=Adv/Prof)",
            "House mortgage in $000",
            "Has securities account (1/0)",
            "Has CD account (1/0)",
            "Uses internet banking (1/0)",
            "Uses Universal Bank credit card (1/0)"
        ]
    })
    st.write(dict_df)

# --------- Tabs ----------
tab1, tab2, tab3 = st.tabs(["📊 Insight Studio (5 Charts)", "🤖 Modeling (DT / RF / GBT)", "🚀 Upload & Predict"])

# ===================== TAB 1: Insights =====================
with tab1:
    st.subheader("📊 Insight Studio — Actionable conversion insights")
    # Ensure target exists
    if TARGET_COL not in df.columns:
        st.warning("Target column not present in uploaded file. Insights will be limited.")
    y = df[TARGET_COL] if TARGET_COL in df.columns else None

    # 1) Heatmap: Conversion rate by Income decile × Family size
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**1) Heatmap – Conversion rate by Income decile × Family size**")
        tmp = df.copy()
        tmp["income_decile"] = pd.qcut(tmp["Income"], 10, labels=False, duplicates='drop')
        pivot = tmp.pivot_table(index="income_decile", columns="Family", values=TARGET_COL, aggfunc="mean")
        fig1 = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns.astype(str), y=pivot.index.astype(str),
                                         colorscale="Viridis", text=np.round(pivot.values,3),
                                         texttemplate="%{text}"))
        fig1.update_layout(xaxis_title="Family size", yaxis_title="Income decile (0=lowest)",
                           title="Higher-income small families convert best?")
        st.plotly_chart(fig1, use_container_width=True)

    # 2) Stacked bars: CCAvg bins × Education split by Online usage (conversion)
    with colB:
        st.markdown("**2) Stacked Bars – Conversion by CCAvg bins × Education (Online=1 vs 0)**")
        tmp = df.copy()
        tmp["cc_bin"] = _bin_series(tmp["CCAvg"], bins=[0,1,2,4,6,20], labels=["≤1","1–2","2–4","4–6",">6"])
        g = tmp.groupby(["cc_bin","Education","Online"])[TARGET_COL].mean().reset_index()
        fig2 = px.bar(g, x="cc_bin", y=TARGET_COL, color="Online", barmode="group",
                      facet_col="Education", category_orders={"cc_bin":["≤1","1–2","2–4","4–6",">6"]},
                      labels={TARGET_COL:"Conversion Rate"}, title="Higher card spend + Online banking → higher conversion")
        st.plotly_chart(fig2, use_container_width=True)

    # 3) Age vs Experience grid: conversion as heatmap
    colC, colD = st.columns(2)
    with colC:
        st.markdown("**3) Heatmap – Conversion by Age × Experience**")
        tmp = df.copy()
        tmp["age_bin"] = _bin_series(tmp["Age"], bins=[20,30,40,50,60,80], labels=["20s","30s","40s","50s","60+"])
        tmp["exp_bin"] = _bin_series(tmp["Experience"], bins=[0,5,10,15,20,40], labels=["0–5","5–10","10–15","15–20","20+"])
        h = tmp.pivot_table(index="age_bin", columns="exp_bin", values=TARGET_COL, aggfunc="mean")
        fig3 = go.Figure(data=go.Heatmap(z=h.values, x=h.columns.astype(str), y=h.index.astype(str),
                                         colorscale="Blues", text=np.round(h.values,3), texttemplate="%{text}"))
        fig3.update_layout(title="Prime segments cluster in 30s–40s with 10–20 yrs experience?",
                           xaxis_title="Experience band", yaxis_title="Age band")
        st.plotly_chart(fig3, use_container_width=True)

    # 4) Correlation matrix of features with target highlight
    with colD:
        st.markdown("**4) Correlation Matrix – Features (incl. target)**")
        corr = df.drop(columns=[ID_COL, ZIP_COL], errors="ignore").corr(numeric_only=True)
        fig4 = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation heatmap")
        st.plotly_chart(fig4, use_container_width=True)

    # 5) Simple propensity model (RF) to produce a Gains chart for targeting strategy
    st.markdown("**5) Targeting Strategy – Cumulative Gains using a quick Random Forest**")
    # Quick train-test
    features = df.drop(columns=[ID_COL, ZIP_COL, TARGET_COL], errors="ignore")
    y = df[TARGET_COL] if TARGET_COL in df.columns else None
    X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    rf_quick = RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=300)
    rf_quick.fit(X_train, y_train)
    proba = rf_quick.predict_proba(X_test)[:,1]
    fig5 = _lift_gain_figure(y_test.to_numpy(), proba)
    st.plotly_chart(fig5, use_container_width=True)
    st.info("**Action hint:** If you contact the **top 20%** scored customers, you capture a large share of likely responders while minimizing spend.")

# ===================== TAB 2: Modeling =====================
with tab2:
    st.subheader("🤖 Train & Compare Models (5-fold CV)")
    st.write("Click **Run Training** to grid-search each model for best AUC (5-fold CV), then compare metrics and plots.")
    features = df.drop(columns=[ID_COL, ZIP_COL, TARGET_COL], errors="ignore")
    target = df[TARGET_COL] if TARGET_COL in df.columns else None

    col = st.columns([1,1,1,2])
    with col[0]:
        test_size = st.slider("Test size", 0.1, 0.4, 0.25, 0.05)
    with col[1]:
        cv = st.slider("CV folds", 3, 10, 5, 1)
    with col[2]:
        rs = st.number_input("Random seed", value=RANDOM_STATE, step=1)

    if st.button("🚀 Run Training", type="primary"):
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, stratify=target, random_state=rs)
        pipelines = _build_pipelines()

        results = {}
        rows = []
        for name, (pipe, grid) in pipelines.items():
            best_est, best_auc_cv, best_params = _fit_with_cv(name, pipe, grid, X_train, y_train, cv=cv)
            res = _evaluate_model(name, best_est, X_train, y_train, X_test, y_test)
            results[name] = res
            rows.append({
                "Algorithm": name,
                "Training Accuracy": round(res["acc_train"],4),
                "Testing Accuracy": round(res["acc_test"],4),
                "Precision": round(res["precision"],4),
                "Recall": round(res["recall"],4),
                "F1-Score": round(res["f1"],4),
                "AUC (Test)": round(res["auc_test"],4),
                "Best AUC (CV)": round(best_auc_cv,4)
            })
        metrics_df = pd.DataFrame(rows).set_index("Algorithm")
        st.success("Training complete. Summary below.")
        st.dataframe(metrics_df, use_container_width=True)

        # ROC combined
        st.plotly_chart(_roc_plotly(results), use_container_width=True)

        # Confusion matrices train & test
        for name, res in results.items():
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(_cm_plotly(res["cm_train"], f"{name} — Confusion Matrix (Train)"), use_container_width=True)
            with c2:
                st.plotly_chart(_cm_plotly(res["cm_test"], f"{name} — Confusion Matrix (Test)"), use_container_width=True)

        # Feature importances
        st.markdown("### Feature Importances")
        cols = st.columns(3)
        for (name, res), container in zip(results.items(), cols*2):  # show all even if >3
            with container:
                st.plotly_chart(_feature_importance_plotly(res["model"], features.columns.tolist(), f"{name} — Feature Importance"),
                                use_container_width=True)

        # Store artifacts for Tab 3 use
        st.session_state["last_models"] = {k: v["model"] for k,v in results.items()}
        st.session_state["last_features"] = features.columns.tolist()

# ===================== TAB 3: Upload & Predict =====================
with tab3:
    st.subheader("🚀 Upload New Data & Predict")
    st.write("Upload a file with the same columns except **Personal Loan** (optional). We'll add predictions and let you download.")

    new_file = st.file_uploader("Upload CSV for scoring", type=["csv"], key="score_file")
    if "last_models" not in st.session_state:
        st.info("Tip: Train models in the **Modeling** tab to use the best model for scoring. Otherwise, a quick Random Forest will be used.")
    scorer_choice = st.selectbox("Model to use", ["Best Random Forest (from Modeling)", "Quick Random Forest (default)"])

    if new_file is not None:
        new_df_raw = pd.read_csv(new_file)
        # Try to clean
        try:
            new_df = _clean_universal_bank(new_df_raw)
        except Exception as e:
            st.error(f"Scoring data error: {e}")
            st.stop()

        # Pick model
        if scorer_choice == "Best Random Forest (from Modeling)" and "last_models" in st.session_state:
            # Find RF model
            rf_model = None
            for k, m in st.session_state["last_models"].items():
                if k.lower().startswith("random forest"):
                    rf_model = m
            if rf_model is None:
                st.warning("No trained Random Forest found in session; falling back to quick RF.")
        else:
            rf_model = None

        # If no trained RF in session, fit a quick one on current df
        if rf_model is None:
            base_features = df.drop(columns=[ID_COL, ZIP_COL, TARGET_COL], errors="ignore")
            base_target = df[TARGET_COL]
            X_tr, X_te, y_tr, y_te = train_test_split(base_features, base_target, test_size=0.25, stratify=base_target, random_state=RANDOM_STATE)
            rf_model = Pipeline([("identity", FunctionTransformer(lambda X: X)),
                                 ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_estimators=300))])
            rf_model.fit(X_tr, y_tr)

        # Ensure feature alignment
        feat_cols = [c for c in new_df.columns if c in REQUIRED_FEATURES]
        X_new = new_df[REQUIRED_FEATURES].copy()
        proba = rf_model.predict_proba(X_new)[:,1]
        pred = (proba >= 0.5).astype(int)

        scored = new_df.copy()
        scored["Pred_Prob"] = proba.round(4)
        scored["Pred_Personal Loan"] = pred

        st.success("Scoring complete.")
        st.dataframe(scored.head(20), use_container_width=True)
        _download_df_button(scored, "scored_personal_loan.csv", "📥 Download scored file")

# ---- Footer ----
st.caption("Built for Streamlit Cloud • No version pins in requirements.txt • DT/RF/GBT with 5-fold CV")
