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
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

print("Chargement du dataset Predictive Maintenance...")
df = pd.read_csv("data/raw/predictive_maintenance.csv")
print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(f"Répartition des classes (Target) :\n{df['Target'].value_counts()}")

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

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("panne_machine_prediction")

preprocesseur = ColumnTransformer(transformers=[
    ("num", StandardScaler(), colonnes_numeriques),
    ("cat", OneHotEncoder(handle_unknown="ignore"), colonnes_categorielles),
])

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
        ("model", RandomForestClassifier(
            class_weight="balanced", 
            random_state=42
        )),
    ]),
}

grille_random_forest = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [5, 10, 15],
    "model__min_samples_split": [2, 5],
}

best_pipeline = None
best_model_name = None
best_f1 = -1

for name, pipeline in pipelines.items():
    with mlflow.start_run(run_name=name):
        print(f"\nEntraînement : {name}")

        if name == "RandomForestClassifier":
            # j'ai mis n_jobs=1 ici sinon ca plante avec pas assez de memoire
            recherche = GridSearchCV(
                pipeline, grille_random_forest,
                scoring="f1", cv=3, n_jobs=1
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

        if f1 > best_f1:
            best_f1 = f1
            best_pipeline = pipeline
            best_model_name = name

# Analyse de l'importance des variables 
if best_model_name == "RandomForestClassifier":
    print("\nCalcul de l'importance des variables...")

    noms_colonnes = (
        colonnes_numeriques
        + list(
            best_pipeline.named_steps["preprocessing"]
            .named_transformers_["cat"]
            .get_feature_names_out(colonnes_categorielles)
        )
    )

    importances = best_pipeline.named_steps["model"].feature_importances_

    indices_tries = importances.argsort()[::-1]
    noms_tries = [noms_colonnes[i] for i in indices_tries]
    valeurs_triees = importances[indices_tries]

    print("Importance des variables (du plus au moins important) :")
    for nom, valeur in zip(noms_tries, valeurs_triees):
        print(f"  {nom} : {valeur:.4f}")

    # Graphique
    plt.figure(figsize=(10, 6))
    plt.barh(noms_tries, valeurs_triees, color="#1F4E78")
    plt.xlabel("Importance")
    plt.title("Importance des variables - RandomForestClassifier")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    chemin_graphique = "feature_importance.png"
    plt.savefig(chemin_graphique)
    plt.close()
    print(f"Graphique sauvegarde : {chemin_graphique}")

    # Log du graphique dans MLflow
    with mlflow.start_run(run_name=f"{best_model_name}_feature_importance"):
        mlflow.log_artifact(chemin_graphique)
        for nom, valeur in zip(noms_tries, valeurs_triees):
            nom_propre = (
                nom.replace("[", "").replace("]", "")
                   .replace(" ", "_")
            )
            mlflow.log_metric(f"importance_{nom_propre}", valeur)            
            
            
print(f"\nMeilleur modèle : {best_model_name} (F1-score = {best_f1:.4f})")

with open("model.pkl", "wb") as f:
    pickle.dump(best_pipeline, f)

print("Pipeline complet sauvegardé dans model.pkl")