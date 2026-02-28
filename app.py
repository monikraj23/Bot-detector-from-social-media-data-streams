import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Bot Detection Dashboard", layout="wide")

st.title("🤖 AI Social Media Bot Detection Dashboard")
st.write("Interactive behavioral analytics system for detecting fake engagement.")

# ---------------- LOAD DATA ----------------
data_path = Path("data/combined_data.csv")   # ✅ FIXED
df = pd.read_csv(data_path)

# ---------------- LOAD MODEL ----------------
model_path = Path("models/bot_model.pkl")
model = joblib.load(model_path)

# ---------------- FEATURE FLAGS ----------------
df["fast_reply_flag"] = (df["reply_delay_seconds"] < 30).astype(int)
df["low_karma_flag"] = (df["user_karma"] < 50).astype(int)
df["new_account_flag"] = (df["account_age_days"] < 30).astype(int)
df["link_spam_flag"] = df["contains_links"].astype(int)
df["neutral_sentiment_flag"] = df["sentiment_score"].between(-0.1, 0.1).astype(int)

# ---------------- ANOMALY DETECTION ----------------
features = [
    "account_age_days","user_karma","reply_delay_seconds",
    "sentiment_score","avg_word_length","contains_links"
]

iso = IsolationForest(contamination=0.1, random_state=42)
df["is_anomaly"] = (iso.fit_predict(df[features]) == -1).astype(int)

# ---------------- CLUSTERING ----------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(
    df[["reply_delay_seconds","user_karma","sentiment_score"]]
)

# ---------------- BOT PREDICTIONS ----------------
model_features = [
    "account_age_days","user_karma","reply_delay_seconds",
    "sentiment_score","avg_word_length","contains_links",
    "bot_probability","fast_reply_flag","low_karma_flag",
    "new_account_flag","link_spam_flag","neutral_sentiment_flag"
]

df["bot_prob"] = model.predict_proba(df[model_features])[:,1]
df["authenticity"] = (1 - df["bot_prob"]) * 100

# ---------------- OVERVIEW METRICS ----------------
st.subheader("📊 System Overview")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Accounts", len(df))
c2.metric("Bots", int(df["is_bot_flag"].sum()))
c3.metric("Anomalies", int(df["is_anomaly"].sum()))
c4.metric("High Risk", int((df["bot_prob"]>0.7).sum()))

st.divider()

# ---------------- LIVE ALERTS ----------------
st.subheader("🚨 Live Alerts")

alerts = df[df["bot_prob"] > 0.85].head(5)

if len(alerts) == 0:
    st.success("No critical threats detected")
else:
    for _, row in alerts.iterrows():
        st.error(f"High-risk bot detected in r/{row['subreddit']} (Score: {row['bot_prob']:.2f})")

st.divider()

# ---------------- FILTER PANEL ----------------
st.subheader("🔎 Filter & Investigate Accounts")

show_bots = st.checkbox("Show Bots Only")
show_anomalies = st.checkbox("Show Anomalies Only")

risk_threshold = st.slider("Minimum Bot Probability", 0.0, 1.0, 0.7, step=0.05)

subs = st.multiselect(
    "Filter by Subreddit",
    options=df["subreddit"].unique(),
    default=df["subreddit"].unique()
)

filtered = df[df["subreddit"].isin(subs)]

if show_bots:
    filtered = filtered[filtered["is_bot_flag"]==1]

if show_anomalies:
    filtered = filtered[filtered["is_anomaly"]==1]

filtered = filtered[filtered["bot_prob"] >= risk_threshold]

st.write(f"Showing {len(filtered)} accounts")

def highlight(val):
    if val > 0.7:
        return "background-color:#ff4b4b;color:white"
    elif val > 0.4:
        return "background-color:#ffa500"
    return ""

st.dataframe(filtered.style.applymap(highlight, subset=["bot_prob"]),
             use_container_width=True)

st.divider()

# ---------------- TOP RISKY ACCOUNTS ----------------
st.subheader("🚨 Top Risky Accounts")
top_risky = df.sort_values("bot_prob", ascending=False).head(10)
st.dataframe(top_risky[["subreddit","bot_prob","authenticity"]])

st.divider()

# ---------------- CLUSTER VISUALIZATION ----------------
st.subheader("📊 Behavioral Cluster Visualization")

fig, ax = plt.subplots()
scatter = ax.scatter(
    df["user_karma"],
    df["reply_delay_seconds"],
    c=df["cluster"]
)
ax.set_xlabel("User Karma")
ax.set_ylabel("Reply Delay")
ax.set_title("Behavior Clusters")

st.pyplot(fig)

st.divider()

# ---------------- FEATURE IMPORTANCE ----------------
st.subheader("🧠 What Influences Bot Detection")

importance_df = pd.DataFrame({
    "Feature": model_features,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))

st.divider()

# ---------------- ACCOUNT INSPECTOR ----------------
st.subheader("🕵️ Inspect an Account")

index = st.number_input("Enter row index", 0, len(df)-1, 0)
row = df.iloc[index]

st.write("### Account Details")
st.write(row)

if row["bot_prob"] > 0.7:
    st.error("⚠ High Risk Bot")
elif row["bot_prob"] > 0.4:
    st.warning("⚠ Medium Risk")
else:
    st.success("✅ Low Risk")

st.write("### Behavioral Triggers")

triggers = []
if row["fast_reply_flag"]: triggers.append("Fast reply speed")
if row["low_karma_flag"]: triggers.append("Low karma")
if row["new_account_flag"]: triggers.append("New account")
if row["link_spam_flag"]: triggers.append("Contains links")
if row["neutral_sentiment_flag"]: triggers.append("Neutral sentiment")
if row["is_anomaly"]: triggers.append("Anomalous behavior")

if triggers:
    for t in triggers:
        st.write("•", t)
else:
    st.write("No suspicious behavior detected")

st.divider()

# ---------------- TEST NEW ACCOUNT ----------------
st.subheader("🔍 Test New Account Behavior")

age = st.slider("Account Age", 1, 2000, 30)
karma = st.slider("User Karma", 0, 10000, 50)
delay = st.slider("Reply Delay", 0, 600, 10)
sent = st.slider("Sentiment", -1.0, 1.0, 0.0)
avg_len = st.slider("Avg Word Length", 1.0, 10.0, 4.5)
links = st.selectbox("Contains Links", [0,1])

fast = 1 if delay < 30 else 0
low = 1 if karma < 50 else 0
new = 1 if age < 30 else 0
neutral = 1 if -0.1 < sent < 0.1 else 0

if st.button("Analyze Behavior"):
    input_data = [[age,karma,delay,sent,avg_len,links,
                   0.5,fast,low,new,links,neutral]]

    prob = model.predict_proba(input_data)[0][1]
    auth = (1-prob)*100

    st.metric("Bot Probability", f"{prob:.2f}")
    st.metric("Authenticity Score", f"{auth:.1f}")

    if prob > 0.7:
        st.error("⚠ High likelihood of bot behavior")
    else:
        st.success("✅ Likely human")

st.divider()
st.caption("Behavioral AI Moderation Dashboard • Hackathon Project")