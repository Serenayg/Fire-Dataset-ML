import argparse
import os
import pandas as pd
from sklearn.externals import joblib  

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.pkl")
    model = joblib.load(model_path)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--output-data-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"),
    )
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
    )

    parser.add_argument("--n_estimators", type=int, default=50)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()


    train_csv_path = os.path.join(args.train, "clean_fire_dataset.csv")
    print(f"📂 Training data path: {train_csv_path}")

    df = pd.read_csv(train_csv_path)

  
    X = df.drop("STATUS", axis=1)
    y = df["STATUS"]


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.random_state,
        stratify=y,
    )

  
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)


    y_pred_test = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    print(f"✅ Test Accuracy: {acc:.3f}")


    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.pkl")
    joblib.dump(model, model_path)

    print(f"💾 Model saved to: {model_path}")