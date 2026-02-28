from feature_engineering import feature_engineering

def detect_coordination():
    df = feature_engineering()

    suspicious = df[
        (df["reply_delay_seconds"] < 15) &
        (df["contains_links"] == 1) &
        (df["account_age_days"] < 20)
    ]

    print("\n⚠ Potential Coordinated Bots:", len(suspicious))

    return suspicious

if __name__ == "__main__":
    detect_coordination()