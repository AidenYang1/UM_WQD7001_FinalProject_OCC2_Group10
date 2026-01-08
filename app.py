
from __future__ import annotations
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st

# ---- Streamlit page config: 必须是首个 Streamlit 调用 ----
st.set_page_config(page_title="GA2 Data Product (Member 4)", layout="wide")

# ---- numpy shim：避免某些 joblib 环境反序列化时找不到模块 ----
sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core._multiarray_umath", np.core._multiarray_umath)

# ---- 如果你们 pipeline 里 pickle 过自定义 transformer，这里必须同名定义 ----
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class ClusterFeatureAdder(BaseEstimator, TransformerMixin):
    def __init__(self, cluster_cols=None, n_clusters=4, random_state=42):
        self.cluster_cols = cluster_cols
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # 反序列化后的对象通常自带 scaler_/kmeans_/cluster_cols_，这里只做兜底
        X_ = X.copy()
        if hasattr(self, "cluster_cols_") and hasattr(self, "scaler_") and hasattr(self, "kmeans_"):
            cols = self.cluster_cols_
            try:
                Xs = self.scaler_.transform(X_[cols])
                X_["cluster_label"] = self.kmeans_.predict(Xs).astype(str)
            except Exception:
                pass
        return X_

import joblib

APP_DIR = Path(__file__).parent
# 支持将产出放在 outputs/ 目录（优先使用 outputs），否则回退当前目录
OUTPUT_DIR = APP_DIR / "outputs"
if OUTPUT_DIR.exists():
    MODEL_PATH = OUTPUT_DIR / "best_classification_model.joblib"
    META_PATH  = OUTPUT_DIR / "best_model_metadata.json"
    SAMPLE_CSV = OUTPUT_DIR / "sample_input_5rows.csv"
else:
    MODEL_PATH = APP_DIR / "best_classification_model.joblib"
    META_PATH  = APP_DIR / "best_model_metadata.json"
    SAMPLE_CSV = APP_DIR / "sample_input_5rows.csv"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_artifacts()

EXPECTED_COLS = meta.get("expected_input_columns", [])
perf = meta.get("test_set_performance", {})

st.title("Heart Disease Risk Predictor — WQD7001_OCC2_Group10_GA2 Data Product")
st.caption("Educational demo only. Not medical advice.")

with st.sidebar:
    threshold = st.slider("Risk threshold", 0.0, 1.0, 0.50, 0.01)
    mode = st.radio("Mode", ["Single patient", "Batch CSV"], index=0)

st.markdown("### Model summary")
st.write(f"Final model: **{meta.get('final_model_name','')}**")
st.write(f"Test set performance: accuracy={perf.get('accuracy')} | recall={perf.get('recall')} | auc={perf.get('auc')}")

def validate(df: pd.DataFrame):
    missing = set(EXPECTED_COLS) - set(df.columns)
    if missing:
        return False, f"Missing columns: {sorted(missing)}"
    return True, ""

def predict(df: pd.DataFrame) -> pd.DataFrame:
    X = df[EXPECTED_COLS].copy()
    # 兜底：某些旧模型/列配置要求 cluster_label；如缺失则补一个默认值，避免 ColumnTransformer 报错
    if "cluster_label" not in X.columns:
        X["cluster_label"] = "0"
    # 分类器通常有 predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    else:
        # 兜底：没有 predict_proba 就用 predict 当作 0/1
        proba = model.predict(X).astype(float)

    pred = (proba >= threshold).astype(int)
    out = df.copy()
    out["risk_probability"] = proba
    out["prediction_label"] = pred
    out["prediction_text"] = np.where(pred == 1, "Yes (HeartDisease)", "No (HeartDisease)")
    return out

if not EXPECTED_COLS:
    st.error("best_model_metadata.json missing expected_input_columns.")
    st.stop()

# 用 sample CSV 给分类字段提供可选值（避免你手打拼写错误）
sample_df = None
if SAMPLE_CSV.exists():
    try:
        sample_df = pd.read_csv(SAMPLE_CSV)
    except Exception:
        sample_df = None

def options_for(col):
    if sample_df is not None and col in sample_df.columns:
        vals = [v for v in sample_df[col].dropna().unique().tolist()]
        vals = [str(v) for v in vals]
        if len(vals) > 0:
            return vals
    return []

NUMERIC_HINT = {"Age","RestingBP","Cholesterol","FastingBS","MaxHR","Oldpeak"}

if mode == "Single patient":
    st.markdown("### Single patient prediction")

    # 让你演示更快：从 sample_input_5rows.csv 选一行自动填充
    preset = {}
    if sample_df is not None:
        idx = st.selectbox("Quick-fill from sample_input_5rows.csv (optional)", list(range(len(sample_df))), index=0)
        if st.button("Load sample row into form"):
            preset = sample_df.iloc[int(idx)].to_dict()
            st.session_state["preset"] = preset
    preset = st.session_state.get("preset", {})

    with st.form("single_form"):
        inputs = {}
        cols = st.columns(3)
        for i, c in enumerate(EXPECTED_COLS):
            with cols[i % 3]:
                if c in NUMERIC_HINT:
                    default_val = float(preset.get(c, 0.0)) if preset else 0.0
                    inputs[c] = st.number_input(c, value=default_val)
                else:
                    opts = options_for(c)
                    default_text = str(preset.get(c, "")) if preset else ""
                    if opts:
                        # 如果 preset 不在 opts 里，仍然允许用户自己改
                        if default_text in opts:
                            inputs[c] = st.selectbox(c, opts, index=opts.index(default_text))
                        else:
                            inputs[c] = st.selectbox(c, opts, index=0)
                    else:
                        inputs[c] = st.text_input(c, value=default_text)

        ok = st.form_submit_button("Predict")

    if ok:
        one = pd.DataFrame([inputs])
        good, msg = validate(one)
        if not good:
            st.error(msg)
        else:
            res = predict(one).iloc[0]
            st.success(f"Prediction: {res['prediction_text']}")
            st.metric("Risk probability", f"{float(res['risk_probability']):.3%}")
            st.dataframe(res.to_frame("value"))

else:
    st.markdown("### Batch CSV prediction")
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        good, msg = validate(df)
        if not good:
            st.error(msg)
        else:
            out = predict(df)
            st.dataframe(out.head(50))
            st.bar_chart(out["risk_probability"])
            st.download_button(
                "Download predictions.csv",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv"
            )
