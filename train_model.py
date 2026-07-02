"""
OptiCrop - Model Training Script
Trains Logistic Regression, KNN, Decision Tree, Random Forest,
compares them, and saves the best model + scaler as .pkl files.
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def main():
    df = pd.read_csv("data/crop_data.csv")
    print(f"Loaded dataset: {df.shape[0]} rows, {df['label'].nunique()} crop classes")

    X = df[FEATURES]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Exploratory clustering (Epic 4, Story 1) - not used for final prediction
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    kmeans.fit(X_train_scaled)
    print("KMeans WCSS (8 clusters):", round(kmeans.inertia_, 2))

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        results[name] = {
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, preds, average="weighted", zero_division=0),
            "F1-Score": f1_score(y_test, preds, average="weighted", zero_division=0),
        }
        trained[name] = model

    results_df = pd.DataFrame(results).T.sort_values("Accuracy", ascending=False)
    print("\nModel comparison:")
    print(results_df.round(4))

    best_name = results_df.index[0]
    best_model = trained[best_name]
    print(f"\nBest model: {best_name}")
    print(classification_report(y_test, best_model.predict(X_test_scaled)))

    with open("model/model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("model/model_name.txt", "w") as f:
        f.write(best_name)

    print("\nSaved model/model.pkl and model/scaler.pkl")


if __name__ == "__main__":
    main()