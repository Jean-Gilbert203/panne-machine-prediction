import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

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

preprocesseur = ColumnTransformer([
    ("num", StandardScaler(), colonnes_numeriques),
    ("cat", OneHotEncoder(handle_unknown="ignore"), colonnes_categorielles),
])

pipeline = Pipeline([
    ("preprocessing", preprocesseur),
    ("model", RandomForestClassifier(random_state=42, class_weight="balanced")),
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print("\nApres ajustement test_size (0.3) et stratify :")
print(classification_report(y_test, predictions, zero_division=0))