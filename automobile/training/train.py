import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Préparer les données
def prepare_data(data):
    data["horsepower"] = pd.to_numeric(data["horsepower"], errors="coerce")
    numeric_df = data.select_dtypes(include=["number"])
    data = data.fillna(numeric_df.mean())
    data = data.drop(["car name"], axis=1)
    return data

# Séparer les données en ensembles de test et d'entraînement
def split_data(df):
    X = df.drop(["mpg"], axis=1).values
    y = df["mpg"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    data = {
        "train": {"X": X_train, "y": y_train},
        "test": {"X": X_test, "y": y_test}
    }
    return data

# Entraîner le modèle et retourner le modèle
def train_model(data, args):
    reg_model = LinearRegression()
    reg_model.fit(data["train"]["X"], data["train"]["y"])
    return reg_model

# Évaluer les métriques du modèle
def get_model_metrics(reg_model, data):
    predictions = reg_model.predict(data["test"]["X"])
    mse = mean_squared_error(data["test"]["y"], predictions)
    return {"mse": mse}

# Fonction principale
def main():
    # Charger les données
    sample_data = pd.read_csv("auto-mpg.csv")

    df = pd.DataFrame(sample_data)
    df = prepare_data(df)

    # Diviser les données en ensembles d'entraînement et de validation
    data = split_data(df)

    # Entraîner le modèle sur l'ensemble d'entraînement
    args = {}
    reg = train_model(data, args)

    # Valider le modèle sur l'ensemble de validation
    metrics = get_model_metrics(reg, data)

    # Sauvegarder le modèle
    model_name = "sklearn_regression_model.pkl"
    joblib.dump(value=reg, filename=model_name)

if __name__ == "__main__":
    main()
