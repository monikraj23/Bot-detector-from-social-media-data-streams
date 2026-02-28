import pandas as pd

real = pd.read_csv("data/reddit_bot_data.csv")
synthetic = pd.read_csv("data/synthetic_data.csv")

combined = pd.concat([real, synthetic], ignore_index=True)

combined.to_csv("data/combined_data.csv", index=False)

print("✅ Combined dataset created")
print("Total records:", len(combined))