import pandas as pd

df = pd.read_csv("data/raw/predictive_maintenance.csv")

print(df["Failure Type"].value_counts())
print()

# on verifie si Target et Failure Type sont toujours coherents
incoherences = df[(df["Target"] == 1) & (df["Failure Type"] == "No Failure")]
print(f"Lignes ou Target=1 mais Failure Type=No Failure : {len(incoherences)}")

incoherences2 = df[(df["Target"] == 0) & (df["Failure Type"] != "No Failure")]
print(f"Lignes ou Target=0 mais Failure Type different de No Failure : {len(incoherences2)}")