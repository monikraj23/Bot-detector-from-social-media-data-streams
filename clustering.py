from feature_engineering import feature_engineering
from sklearn.cluster import KMeans

def cluster_behavior():
    df = feature_engineering()

    features = [
        "reply_delay_seconds",
        "user_karma",
        "sentiment_score"
    ]

    kmeans = KMeans(n_clusters=3, random_state=42)
    df["behavior_cluster"] = kmeans.fit_predict(df[features])

    print(df["behavior_cluster"].value_counts())

    return df

if __name__ == "__main__":
    cluster_behavior()