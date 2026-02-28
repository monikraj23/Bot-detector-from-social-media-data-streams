import joblib
import pandas as pd
from feature_engineering import feature_engineering

def predict():
    df = feature_engineering()

    model = joblib.load("../models/bot_model.pkl")

    features = [
        "account_age_days",
        "user_karma",
        "reply_delay_seconds",
        "sentiment_score",
        "avg_word_length",
        "contains_links",
        "bot_probability",
        "fast_reply_flag",
        "low_karma_flag",
        "new_account_flag",
        "link_spam_flag",
        "neutral_sentiment_flag"
    ]

    df["predicted_bot"] = model.predict(df[features])
    df["bot_probability_score"] = model.predict_proba(df[features])[:,1]

    # authenticity score
    df["authenticity_score"] = (1 - df["bot_probability_score"]) * 100

    print("\n🔍 Sample Predictions\n")
    print(df[[
        "subreddit",
        "bot_probability_score",
        "authenticity_score",
        "predicted_bot"
    ]].head())

    return df


if __name__ == "__main__":
    predict()