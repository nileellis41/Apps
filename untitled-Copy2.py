import os, pandas as pd, textwrap
import os
import io
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

APP_TITLE = "AccessAlpha — Project Progress Tracker"

# ---------------------------
# Page & Theme Configuration
# ---------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🧭", layout="wide")

# Minimal dark look for charts
plt.rcParams["figure.facecolor"] = "black"
plt.rcParams["axes.facecolor"] = "black"
plt.rcParams["savefig.facecolor"] = "black"
plt.rcParams["axes.edgecolor"] = "white"
plt.rcParams["axes.labelcolor"] = "white"
plt.rcParams["xtick.color"] = "white"
plt.rcParams["ytick.color"] = "white"
plt.rcParams["text.color"] = "white"
plt.rcParams["grid.color"] = "gray"

STATUSES = ["Not started", "In progress", "Blocked", "Done"]
PRIORITIES = ["P0", "P1", "P2", "P3"]

# ---------------------------
# Helpers
# ---------------------------
def starter_tracker() -> pd.DataFrame:
    return pd.DataFrame(columns=["Area","Subtask","Status","Owner","Priority","ETA","Dependencies","Notes","% Complete"])

def load_from_upload(file) -> pd.DataFrame:
    if file is None:
        return None
    try:
        df = pd.read_csv(file)
        if "% Complete" in df.columns:
            df["% Complete"] = pd.to_numeric(df["% Complete"], errors="coerce").fillna(0).astype(int)
        return df
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        return None

def area_progress(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["Area", "Progress"])
    grp = df.groupby("Area")["% Complete"].mean().reset_index()
    grp["Progress"] = (grp["% Complete"] / 100.0).round(2)
    return grp[["Area","Progress"]]

def progress_ring(pct: float, title: str = "", size=(2.2,2.2)):
    fig, ax = plt.subplots(figsize=size)
    ax.axis("equal")
    ax.pie([pct, 1-pct], startangle=90, counterclock=False,
           wedgeprops=dict(width=0.3), labels=["",""])
    ax.set_facecolor("black")
    ax.text(0, 0, f"{int(pct*100)}%", ha="center", va="center", fontsize=16, color="white")
    if title:
        ax.set_title(title, color="white", fontsize=10)
    st.pyplot(fig)

def status_badge(s: str) -> str:
    return {
        "Not started": "🕒 Not started",
        "In progress": "⚙️ In progress",
        "Blocked": "⛔ Blocked",
        "Done": "✅ Done",
    }.get(s, s)

def celebrate_if_done(old_df: pd.DataFrame, new_df: pd.DataFrame):
    if old_df is None or new_df is None or old_df.empty or new_df.empty:
        return
    merged = new_df.merge(old_df, how="left", left_index=True, right_index=True, suffixes=("", "_old"))
    if "Status_old" in merged.columns:
        just_completed = (merged["Status"] == "Done") & (merged["Status_old"] != "Done")
        if just_completed.any():
            st.balloons()
            st.toast("Nice! Another task knocked out. 💥")

# ---------------------------
# Sidebar: Upload & Filters
# ---------------------------
st.sidebar.title("🎛️ Controls")

uploaded = st.sidebar.file_uploader("Upload your tracker CSV", type=["csv"])

# Initialize session df
if "tracker_df" not in st.session_state:
    st.session_state["tracker_df"] = None

if uploaded:
    st.session_state["tracker_df"] = load_from_upload(uploaded)
elif st.session_state["tracker_df"] is None:
    st.session_state["tracker_df"] = starter_tracker()

df = st.session_state["tracker_df"]

st.sidebar.markdown("**Quick Filters**")
area_filter = st.sidebar.multiselect("Area", sorted(df["Area"].dropna().unique().tolist()) if not df.empty else [])
status_filter = st.sidebar.multiselect("Status", STATUSES, default=STATUSES)
prio_filter = st.sidebar.multiselect("Priority", PRIORITIES, default=PRIORITIES)
owner_filter = st.sidebar.text_input("Owner contains…", value="")

st.sidebar.markdown("---")
st.sidebar.markdown("**Quick Add Task**")
with st.sidebar.form("quick_add"):
    qa_area_opts = sorted(df["Area"].dropna().unique().tolist()) if not df.empty else ["Overview"]
    qa_area = st.selectbox("Area", qa_area_opts)
    qa_sub = st.text_input("Subtask")
    qa_owner = st.text_input("Owner")
    qa_prio = st.selectbox("Priority", PRIORITIES, index=1)
    qa_eta = st.date_input("ETA")
    submitted = st.form_submit_button("➕ Add")
    if submitted and qa_sub:
        new_row = {
            "Area": qa_area, "Subtask": qa_sub, "Status": "Not started",
            "Owner": qa_owner, "Priority": qa_prio, "ETA": qa_eta.isoformat(),
            "Dependencies": "", "Notes": "", "% Complete": 0
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        st.session_state["tracker_df"] = df
        st.toast("Task added!")

# Apply filters
fdf = df.copy()
if area_filter:
    fdf = fdf[fdf["Area"].isin(area_filter)]
if status_filter:
    fdf = fdf[fdf["Status"].isin(status_filter)]
if prio_filter:
    fdf = fdf[fdf["Priority"].isin(prio_filter)]
if owner_filter:
    fdf = fdf[fdf["Owner"].fillna("").str.contains(owner_filter, case=False)]

# ---------------------------
# Header & KPIs
# ---------------------------
st.title("🧭 AccessAlpha — Creative Progress Tracker")
colA, colB, colC, colD = st.columns(4)
total = len(df)
done = (df["Status"] == "Done").sum() if total else 0
pct_all = int(df["% Complete"].mean()) if total else 0
blocked = (df["Status"] == "Blocked").sum() if total else 0
colA.metric("Total tasks", total)
colB.metric("Done", done)
colC.metric("Blocked", blocked)
colD.metric("Overall %", f"{pct_all}%")

# Area progress rings
st.markdown("### Area Progress")
ap = area_progress(df)
if ap.empty:
    st.info("No tasks yet. Upload a CSV or add tasks from the sidebar!")
else:
    cols = st.columns(min(6, len(ap)))
    for i, (_, row) in enumerate(ap.iterrows()):
        with cols[i % len(cols)]:
            progress_ring(row["Progress"], title=row["Area"][:22])

st.markdown("---")

# ---------------------------
# Editable Grid
# ---------------------------
st.subheader("📝 Update Tasks Inline")
config = {
    "Status": st.column_config.SelectboxColumn("Status", options=STATUSES, help="Current status"),
    "Priority": st.column_config.SelectboxColumn("Priority", options=PRIORITIES, help="Priority"),
    "% Complete": st.column_config.NumberColumn("% Complete", min_value=0, max_value=100, step=5, format="%d"),
    "ETA": st.column_config.DateColumn("ETA"),
    "Owner": st.column_config.TextColumn("Owner"),
    "Dependencies": st.column_config.TextColumn("Dependencies"),
    "Notes": st.column_config.TextColumn("Notes"),
}

before = fdf.copy()
edited = st.data_editor(
    fdf,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config=config,
    key="editor",
)

col1, col2, col3 = st.columns([1,1,2])
with col1:
    if st.button("💾 Save to Session"):
        # Persist edited filtered view back into full df by index mapping
        df_update = df.copy()
        # Rebuild mapping using positions: safe because we didn't shuffle rows
        # Align columns
        common_cols = [c for c in df_update.columns if c in edited.columns]
        df_update.loc[fdf.index, common_cols] = edited[common_cols].values
        df_update["% Complete"] = pd.to_numeric(df_update["% Complete"], errors="coerce").fillna(0).astype(int)
        celebrate_if_done(before, edited)
        st.session_state["tracker_df"] = df_update
        st.success("Saved changes in memory (session).")
with col2:
    csv_bytes = st.session_state["tracker_df"].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="AccessAlpha_Progress_Tracker.csv", mime="text/csv")
with col3:
    st.caption("Tip: Change **Status** to *Done* to trigger 🎈 confetti.")

st.markdown("---")

# ---------------------------
# Kanban View (by Status)
# ---------------------------
st.subheader("🗂️ Kanban Snapshot")
kan_cols = st.columns(4)
for i, s in enumerate(STATUSES):
    with kan_cols[i]:
        st.markdown(f"**{status_badge(s)}**")
        kdf = fdf[fdf["Status"] == s][["Area","Subtask","Owner","Priority","% Complete","ETA"]]
        if kdf.empty:
            st.write("_No tasks_")
        else:
            for _, r in kdf.iterrows():
                st.markdown(
                    f"- **{r['Subtask']}** • {r['Area']} • {r['Priority']} • {int(r['% Complete'])}% • {r.get('Owner','') or 'Unassigned'} • _ETA: {r.get('ETA','')}_"
                )

# ---------------------------
# Focus Pane for Quick Edits
# ---------------------------
st.markdown("---")
st.subheader("🎯 Focus Pane")
if not fdf.empty:
    idx = st.selectbox("Pick a task to focus", options=range(len(fdf)), format_func=lambda i: f"[{fdf.iloc[i]['Area']}] {fdf.iloc[i]['Subtask']}")
    task = fdf.iloc[idx].copy()
    c1, c2, c3 = st.columns(3)
    with c1:
        task["Status"] = st.selectbox("Status", STATUSES, index=STATUSES.index(task["Status"]))
    with c2:
        task["Priority"] = st.selectbox("Priority", PRIORITIES, index=PRIORITIES.index(task["Priority"]))
    with c3:
        task["% Complete"] = st.slider("% Complete", 0, 100, int(task["% Complete"]), step=5)

    task["Owner"] = st.text_input("Owner", value=str(task.get("Owner","")))
    task["Notes"] = st.text_area("Notes", value=str(task.get("Notes","")), height=100)

    if st.button("✅ Apply to Selected Task"):
        fdf.iloc[idx] = task
        df_commit = st.session_state["tracker_df"].copy()
        df_commit.loc[fdf.index[idx], fdf.columns] = task.values
        df_commit["% Complete"] = pd.to_numeric(df_commit["% Complete"], errors="coerce").fillna(0).astype(int)
        celebrate_if_done(df.loc[[fdf.index[idx]]], df_commit.loc[[fdf.index[idx]]])
        st.session_state["tracker_df"] = df_commit
        st.success("Task updated 🎉")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")