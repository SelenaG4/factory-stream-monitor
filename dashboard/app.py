"""Factory Stream Monitor -- real-time operations dashboard.

Reads the streaming job's committed outputs (window metrics, model alerts, drift log,
grounded incident reports) and plays them back on a moving time cursor with auto-refresh,
so throughput, alerts, and drift build up live -- the monitoring view a plant operator
would watch. In production the same page tails the live sinks the Spark job writes.
"""
from __future__ import annotations

import json
from glob import glob
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "stream_out"

# dataviz palette
SITE_COLORS = {"Zurich": "#2a78d6", "Basel": "#eb6834", "Geneva": "#1baf7a"}
CRIT, WARN, GOOD = "#d03b3b", "#fab219", "#0ca30c"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
SEV_COLOR = {"critical": CRIT, "warning": WARN, "watch": "#eb6834"}

st.set_page_config(page_title="Factory Stream Monitor", page_icon="📡", layout="wide")


@st.cache_data
def load_all():
    wm = pd.concat([pd.read_parquet(f) for f in glob(str(OUT / "window_metrics" / "*.parquet"))],
                   ignore_index=True) if glob(str(OUT / "window_metrics" / "*.parquet")) else pd.DataFrame()
    if not wm.empty:
        wm["window_start"] = pd.to_datetime(wm["window_start"])
    alerts = pd.DataFrame([json.loads(l) for l in open(OUT / "alerts.jsonl")]) if (OUT / "alerts.jsonl").exists() else pd.DataFrame()
    if not alerts.empty:
        alerts["event_time"] = pd.to_datetime(alerts["event_time"])
    drift = pd.DataFrame([json.loads(l) for l in open(OUT / "drift.jsonl")]) if (OUT / "drift.jsonl").exists() else pd.DataFrame()
    incidents = json.loads((OUT / "incidents.json").read_text()) if (OUT / "incidents.json").exists() else []
    summary = json.loads((OUT / "run_summary.json").read_text()) if (OUT / "run_summary.json").exists() else {}
    model = json.loads((ROOT / "data" / "model_metrics.json").read_text()) if (ROOT / "data" / "model_metrics.json").exists() else {}
    return wm, alerts, drift, incidents, summary, model


def base(fig, height=300, ylab=""):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=26, b=8),
                      paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK, size=13),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(showgrid=False, linecolor="#c3c2b7", tickcolor=MUTED)
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, title_text=ylab, title_font=dict(color=MUTED))
    return fig


wm, alerts, drift, incidents, summary, model = load_all()

# ---- playback cursor + auto-refresh (advances over the run, then loops) ----
windows = sorted(wm["window_start"].unique()) if not wm.empty else []
n_win = len(windows)
live = st.sidebar.toggle("▶ Live playback", value=False)
if live and n_win:
    try:
        from streamlit_autorefresh import st_autorefresh
        tick = st_autorefresh(interval=1500, key="tick")
    except Exception:
        tick = st.session_state.get("tick", 0) + 1
        st.session_state["tick"] = tick
    cursor = tick % n_win + 1
else:
    cursor = st.sidebar.slider("Time cursor (window #)", 1, max(n_win, 1), max(n_win, 1)) if n_win else 1

now = windows[cursor - 1] if n_win else None
wm_v = wm[wm["window_start"] <= now] if n_win else wm
al_v = alerts[alerts["event_time"] <= now] if (n_win and not alerts.empty) else alerts
frac = cursor / n_win if n_win else 1.0
dr_v = drift.iloc[: max(1, round(len(drift) * frac))] if not drift.empty else drift

st.title("📡 Factory Stream Monitor")
st.caption("Real-time anomaly detection on streaming factory telemetry — Spark Structured Streaming + an "
           "MLlib model, with drift monitoring and LLM-generated incident reports. "
           f"Playback: window {cursor}/{n_win}" + (f" · {pd.Timestamp(now):%H:%M}" if now is not None else ""))

# ---- KPI row ----
events_seen = int(wm_v["events"].sum()) if not wm_v.empty else 0
n_alerts = len(al_v)
rate = n_alerts / events_seen * 100 if events_seen else 0
drifted = int(dr_v["n_drifted_sensors"].iloc[-1]) if not dr_v.empty else 0
k1, k2, k3, k4 = st.columns(4)
k1.metric("Events processed", f"{events_seen:,}")
k2.metric("Anomaly alerts", f"{n_alerts:,}", f"{rate:.1f}% of stream")
k3.metric("Model AUC (held-out)", f"{model.get('auc', 0):.3f}",
          f"P {model.get('precision',0):.2f} / R {model.get('recall',0):.2f}")
k4.metric("Drift monitor", "DRIFT" if drifted else "stable",
          f"{drifted} sensor(s) shifted", delta_color="inverse")

st.divider()
c1, c2 = st.columns([3, 2])
with c1:
    st.subheader("Throughput — events per minute, by site")
    fig = go.Figure()
    if not wm_v.empty:
        for site, g in wm_v.groupby("site"):
            gg = g.groupby("window_start", as_index=False)["events"].sum()
            fig.add_trace(go.Scatter(x=gg["window_start"], y=gg["events"], name=site, mode="lines",
                                     line=dict(color=SITE_COLORS.get(site, "#4a3aa7"), width=2)))
    fig.update_xaxes(title_text="event time")
    st.plotly_chart(base(fig, ylab="events / min"), width="stretch")
with c2:
    st.subheader("Feature drift")
    fig = go.Figure()
    if not dr_v.empty:
        fig.add_trace(go.Scatter(x=dr_v["batch_id"], y=dr_v["max_drift_z"], mode="lines+markers",
                                 line=dict(color=WARN, width=2), name="max drift (σ)"))
        fig.add_hline(y=1.0, line=dict(color=CRIT, width=1, dash="dash"))
    fig.update_xaxes(title_text="micro-batch")
    st.plotly_chart(base(fig, ylab="max |mean shift| (σ)"), width="stretch")
    st.caption("Dashed line = drift threshold (1σ). The tail of the run carries an injected "
               "fleet-wide temperature/vibration drift — the monitor catches it.")

st.divider()
c3, c4 = st.columns([2, 3])
with c3:
    st.subheader("Alerts by machine type")
    if not al_v.empty:
        by = al_v["machine_type"].value_counts()
        fig = go.Figure(go.Bar(x=by.values, y=[m.replace("_", " ") for m in by.index],
                               orientation="h", marker_color="#2a78d6", marker_line_width=0))
        st.plotly_chart(base(fig, height=260, ylab=""), width="stretch")
    st.metric("True anomaly rate (synthetic ground truth)", f"{summary.get('true_anomaly_rate_pct', 0)}%")
with c4:
    st.subheader("🛠 GenAI incident reports (grounded in the maintenance KB)")
    shown = [i for i in incidents if i["machine_id"] in set(al_v["machine_id"])] if not al_v.empty else []
    shown = sorted(shown, key=lambda x: -x["anomaly_prob"])[:4]
    for i in shown:
        color = SEV_COLOR.get(i["severity"], MUTED)
        st.markdown(
            f"<div style='border-left:4px solid {color};padding:6px 12px;margin-bottom:8px;background:#f7f7f5'>"
            f"<b>{i['machine_id']}</b> · {i['site']} · <span style='color:{color};font-weight:600'>"
            f"{i['severity'].upper()}</span> (p={i['anomaly_prob']})<br>"
            f"<small><b>Symptom:</b> {i['symptom']}<br>"
            f"<b>Likely cause:</b> {i['likely_cause']} · <b>source:</b> {i['source']}<br>"
            f"<b>Action:</b> {i['recommended_action'][:160]}…</small></div>",
            unsafe_allow_html=True)

st.divider()
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("MLlib model", model.get("model", "GBT").split(" (")[0])
mc2.metric("Precision / Recall", f"{model.get('precision',0):.2f} / {model.get('recall',0):.2f}")
mc3.metric("MLflow registry", f"{model.get('registry_model','—')} v{model.get('registry_version','?')}")
mc4.metric("Sites × machines", f"{len(summary.get('sites', []))} × {summary.get('machines', 0)}")
st.caption("Stream: Spark Structured Streaming (watermarked 1-min windows) · Model: Spark MLlib GBT "
           "Pipeline (registered in MLflow) · Incidents: TF-IDF retrieval over maintenance guides, "
           "grounded structured output · Source: Kafka (prod) / file (this demo).")
