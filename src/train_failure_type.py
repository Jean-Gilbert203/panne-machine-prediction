import pickle
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/raw/predictive_maintenance.csv")

print(df["Failure Type"].value_counts())
print()

# on verifie si Target et Failure Type sont toujours coherents
incoherences = df[(df["Target"] == 1) & (df["Failure Type"] == "No Failure")]
print(f"Lignes ou Target=1 mais Failure Type=No Failure : {len(incoherences)}")

incoherences2 = df[(df["Target"] == 0) & (df["Failure Type"] != "No Failure")]
print(f"Lignes ou Target=0 mais Failure Type different de No Failure : {len(incoherences2)}")

# on garde que les vraies pannes, pas No Failure
pannes = df[df["Failure Type"] != "No Failure"].copy()
print(f"\nNombre de pannes avec un type connu : {len(pannes)}")

colonnes_numeriques = [
    "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"
]
colonnes_categorielles = ["Type"]

X = pannes[colonnes_numeriques + colonnes_categorielles]
y = pannes["Failure Type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("panne_machine_type_prediction")

preprocesseur = ColumnTransformer([
    ("num", StandardScaler(), colonnes_numeriques),
    ("cat", OneHotEncoder(handle_unknown="ignore"), colonnes_categorielles),
])

pipeline = Pipeline([
    ("preprocessing", preprocesseur),
    ("model", RandomForestClassifier(
        n_estimators=150, random_state=42, class_weight="balanced"
    )),
])

with mlflow.start_run(run_name="RandomForest_type_panne_v2"):
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    
    rapport = classification_report(y_test, predictions, zero_division=0)
    print(rapport)
    print("Matrice de confusion :")
    print(confusion_matrix(y_test, predictions))

    mlflow.log_param("model_type", "RandomForestClassifier_multiclasse")
    mlflow.log_param("n_estimators", 150)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_param("test_size", 0.3)
    mlflow.log_text(rapport, "classification_report.txt")
    mlflow.sklearn.log_model(pipeline, "model_type_panne")
    
with open("model_failure_type.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("\nModele de type de panne sauvegarde dans model_failure_type.pkl")
print(classification_report(y_test, predictions, zero_division=0))