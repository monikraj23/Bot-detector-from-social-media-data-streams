from load_data import load_data

def feature_engineering():
    df = load_data()

    # suspiciously fast replies (bots reply quickly)
    df["fast_reply_flag"] = df["reply_delay_seconds"].apply(
        lambda x: 1 if x < 30 else 0
    )

    # low karma indicator
    df["low_karma_flag"] = df["user_karma"].apply(
        lambda x: 1 if x < 50 else 0
    )

    # new account indicator
    df["new_account_flag"] = df["account_age_days"].apply(
        lambda x: 1 if x < 30 else 0
    )

    # excessive links indicator
    df["link_spam_flag"] = df["contains_links"].astype(int)

    # neutral sentiment bots often show low emotional variation
    df["neutral_sentiment_flag"] = df["sentiment_score"].apply(
        lambda x: 1 if -0.1 < x < 0.1 else 0
    )

    print("\n✅ Feature engineering complete\n")
    print(df.head())

    return df


if __name__ == "__main__":
    feature_engineering()