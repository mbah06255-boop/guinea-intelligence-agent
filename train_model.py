import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib

np.random.seed(42)
n = 6000

# ---- Génération de données synthétiques (profils de microfinance Guinée) ----
revenu = np.random.lognormal(mean=14.5, sigma=0.5, size=n)       # revenu mensuel en GNF
dettes = revenu * np.random.uniform(0, 0.6, size=n)              # dettes mensuelles
epargne = revenu * np.random.uniform(0, 0.3, size=n)             # épargne mensuelle
anciennete = np.random.randint(1, 120, size=n)                   # ancienneté en mois

debt_ratio = dettes / revenu
savings_ratio = epargne / revenu

# Un profil "risqué" a un ratio d'endettement élevé, peu d'épargne, peu d'ancienneté
risk_score = (
    debt_ratio * 3
    - savings_ratio * 2
    - (anciennete / 120) * 1.5
    + np.random.normal(0, 0.3, size=n)  # bruit réaliste
)
default_proba_true = 1 / (1 + np.exp(-(risk_score - 0.5) * 4))
default = (np.random.rand(n) < default_proba_true).astype(int)

df = pd.DataFrame({
    "revenu_mensuel_gnf": revenu,
    "dettes_mensuelles_gnf": dettes,
    "epargne_mensuelle_gnf": epargne,
    "anciennete_activite_mois": anciennete,
    "default": default
})

features = ["revenu_mensuel_gnf", "dettes_mensuelles_gnf", "epargne_mensuelle_gnf", "anciennete_activite_mois"]
X = df[features]
y = df["default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

preds = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, preds)
print(f"AUC sur le jeu de test : {auc:.3f}")

os.makedirs("app/models", exist_ok=True)
joblib.dump(model, "app/models/credit_model.pkl")
print("✅ Modèle sauvegardé dans app/models/credit_model.pkl")