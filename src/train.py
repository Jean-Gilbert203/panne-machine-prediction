"""
Script d'entraînement du modèle de prédiction de panne machine (maintenance prédictive).
Compare LogisticRegression et RandomForestClassifier et sauvegarde le meilleur
pipeline complet en model.pkl.
"""

import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# --- 1. Chargement des données ---
print("Chargement du dataset Predictive Maintenance...")
df = pd.read_csv("data/raw/predictive_maintenance.csv")
print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(f"Répartition des classes (Target) :\n{df['Target'].value_counts()}")

# --- 2. Préparation des données ---
# On exclut UDI, Product ID (identifiants) et Failure Type (fuite de données)
X = df.drop(columns=["UDI", "Product ID", "Target", "Failure Type"])
y = df["Target"]

colonnes_numeriques = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
]
colonnes_categorielles = ["Type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 3. Configuration MLflow ---
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("panne_machine_prediction")

# --- 4. Préprocesseur commun---
preprocesseur = ColumnTransformer(transformers=[
    ("num", StandardScaler(), colonnes_numeriques),
    ("cat", OneHotEncoder(handle_unknown="ignore"), colonnes_categorielles),
])

# --- 5. Définition des pipelines (preprocessing + modèle) ---
pipelines = {
    "LogisticRegression": Pipeline([
        ("preprocessing", preprocesseur),
        ("model", LogisticRegression(
            class_weight="balanced", 
            max_iter=1000
        )),
    ]),
    "RandomForestClassifier": Pipeline([
        ("preprocessing", preprocesseur),
        ("model", RandomForestClassifier(class_weight="balanced", random_state=42)),
    ]),
}

# Grille d'hyperparametres a tester uniquement pour RandomForest
grille_random_forest = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 10, 15],
    "model__min_samples_split": [2, 5],
}

best_pipeline = None
best_model_name = None
best_f1 = -1

# --- 6. Entraînement et comparaison des pipelines ---
for name, pipeline in pipelines.items():
    with mlflow.start_run(run_name=name):
        print(f"\nEntraînement : {name}")

        if name == "RandomForestClassifier":
            recherche = GridSearchCV(
                pipeline, grille_random_forest,
                scoring="f1", cv=3, n_jobs=-1
            )
            recherche.fit(X_train, y_train)
            pipeline = recherche.best_estimator_
            print(f"  Meilleurs hyperparametres trouves : {recherche.best_params_}")
            mlflow.log_params(recherche.best_params_)
        else:
            pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        acc = accuracy_score(y_test, predictions)
        prec = precision_score(y_test, predictions)
        rec = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)

        print(f"  Accuracy  : {acc:.4f}")
        print(f"  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1-score  : {f1:.4f}")
        print(f"  Matrice de confusion :\n{cm}")

        # Log dans MLflow
        mlflow.log_param("model_type", name)
        mlflow.log_param("preprocessing", "StandardScaler + OneHotEncoder")
        mlflow.log_param("class_weight", "balanced")
        if name == "RandomForestClassifier":
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(pipeline, name)

        # On sélectionne le meilleur modèle selon le F1-score
        if f1 > best_f1:
            best_f1 = f1
            best_pipeline = pipeline
            best_model_name = name

# --- 7. Sauvegarde du meilleur pipeline complet ---
print(f"\nMeilleur modèle : {best_model_name} (F1-score = {best_f1:.4f})")

with open("model.pkl", "wb") as f:
    pickle.dump(best_pipeline, f)

print("Pipeline complet sauvegardé dans model.pkl")