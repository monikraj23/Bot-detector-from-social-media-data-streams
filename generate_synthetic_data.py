import pandas as pd
import numpy as np

np.random.seed(42)

def generate_synthetic_data(n=10000):

    data = {
        "comment_id": [f"syn_{i}" for i in range(n)],
        "subreddit": np.random.choice(
            ["funny","gaming","politics","technology","pics","worldnews"], n
        ),
        "account_age_days": np.random.randint(1, 2000, n),
        "user_karma": np.random.randint(0, 10000, n),
        "reply_delay_seconds": np.random.randint(1, 600, n),
        "sentiment_score": np.random.uniform(-1, 1, n),
        "avg_word_length": np.random.uniform(3, 7, n),
        "contains_links": np.random.choice([0,1], n, p=[0.8,0.2]),
    }

    df = pd.DataFrame(data)

    # simulate bot behavior patterns
    bot_mask = np.random.choice([0,1], size=n, p=[0.7,0.3])

    df["is_bot_flag"] = bot_mask

    # bots behave differently
    df.loc[bot_mask==1, "reply_delay_seconds"] = np.random.randint(1, 25, bot_mask.sum())
    df.loc[bot_mask==1, "user_karma"] = np.random.randint(0, 50, bot_mask.sum())
    df.loc[bot_mask==1, "account_age_days"] = np.random.randint(1, 30, bot_mask.sum())
    df.loc[bot_mask==1, "contains_links"] = 1

    df["bot_type_label"] = df["is_bot_flag"].apply(
        lambda x: "synthetic_bot" if x==1 else "synthetic_human"
    )

    df["bot_probability"] = np.where(
        df["is_bot_flag"]==1,
        np.random.uniform(0.7,1.0,n),
        np.random.uniform(0.0,0.3,n)
    )

    df.to_csv("data/synthetic_data.csv", index=False)

    print("✅ Synthetic dataset created: data/synthetic_data.csv")
    print(df.head())

if __name__ == "__main__":
    generate_synthetic_data()