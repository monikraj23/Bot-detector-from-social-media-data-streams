from feature_engineering import feature_engineering
from sklearn.ensemble import IsolationForest

def detect_anomalies():
    df = feature_engineering()

    features = [
        "account_age_days",
        "user_karma",
        "reply_delay_seconds",
        "sentiment_score",
        "avg_word_length",
        "contains_links"
    ]

    model = IsolationForest(contamination=0.1, random_state=42)
    df["anomaly_score"] = model.fit_predict(df[features])

    # -1 = anomaly
    df["is_anomaly"] = df["anomaly_score"].apply(lambda x: 1 if x == -1 else 0)

    print(df[["is_anomaly"]].value_counts())

    return df

if __name__ == "__main__":
    detect_anomalies()