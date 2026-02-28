from feature_engineering import feature_engineering
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path

def train_model():
    df = feature_engineering()

    # target label
    y = df["is_bot_flag"]

    # features used
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

    X = df[features]

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # train model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    # evaluate
    preds = model.predict(X_test)

    print("\n📊 Model Performance\n")
    print(classification_report(y_test, preds))
    print("Accuracy:", accuracy_score(y_test, preds))

    # save model to correct location
    model_path = Path(__file__).resolve().parent.parent / "models" / "bot_model.pkl"
    joblib.dump(model, model_path)

    print(f"\n✅ Model saved at: {model_path}")

if __name__ == "__main__":
    train_model()