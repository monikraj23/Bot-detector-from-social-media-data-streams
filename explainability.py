import pandas as pd
from predict import predict

def generate_explanations():
    df = predict()

    explanations = []

    for _, row in df.iterrows():
        reasons = []

        if row["fast_reply_flag"] == 1:
            reasons.append("⚠ Replies unusually fast")

        if row["low_karma_flag"] == 1:
            reasons.append("⚠ Low karma account")

        if row["new_account_flag"] == 1:
            reasons.append("⚠ Newly created account")

        if row["link_spam_flag"] == 1:
            reasons.append("⚠ Contains spam links")

        if row["neutral_sentiment_flag"] == 1:
            reasons.append("⚠ Neutral sentiment pattern")

        if row["bot_probability_score"] > 0.8:
            reasons.append("⚠ High bot probability score")

        if not reasons:
            reasons.append("✅ Behavior appears normal")

        explanations.append("; ".join(reasons))

    df["explanation"] = explanations

    print("\n🧠 Detection Explanations\n")
    print(df[[
        "subreddit",
        "bot_probability_score",
        "predicted_bot",
        "explanation"
    ]].head())

    return df


if __name__ == "__main__":
    generate_explanations()