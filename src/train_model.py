import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def load_data(path="data/shopping_data.csv"):
    df = pd.read_csv(path)
    return df

def build_pipeline(df: pd.DataFrame) -> Pipeline:
    target_col = "shopping_time_min"

    numeric_features = [
        "age",
        "hour",
        "day_of_week",
        "total_items",
        "nb_categories",
        "items_alimentaire",
        "items_vetements",
        "items_electronique",
        "items_maison",
        "items_beaute",
        "items_sport",
        "items_librairie",
    ]

    binary_features = [
        "is_weekend",
        "is_sales",
        "is_holiday",
        "has_shopping_list",
    ]

    categorical_features = [
        "gender",
        "profile",
        "store_type",
        "period",
        "special_event",
    ]

    X = df[numeric_features + binary_features + categorical_features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    binary_transformer = "passthrough"
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("bin", binary_transformer, binary_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Évaluation sur le jeu de test :")
    print(f"  RMSE : {rmse:.2f} minutes")
    print(f"  MAE  : {mae:.2f} minutes")
    print(f"  R²   : {r2:.3f}")

    return pipeline

def main():
    print("Entraînement du modèle de prédiction du temps de shopping...")
    df = load_data()

    pipeline = build_pipeline(df)

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "shopping_time_model.joblib")
    joblib.dump(pipeline, model_path)

    print(f"Modèle sauvegardé dans : {model_path}")

if __name__ == "__main__":
    main()
