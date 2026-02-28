import pandas as pd

def load_data():
    df = pd.read_csv("data/reddit_bot_data.csv")

    print("Original Shape:", df.shape)
    print("Columns:", df.columns)

    # drop missing text
    df = df.dropna()


    return df

if __name__ == "__main__":
    df = load_data()
    print(df.head())