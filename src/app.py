from pathlib import Path

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


DATA_URL = (
    "https://raw.githubusercontent.com/4GeeksAcademy/"
    "linear-regression-project-tutorial/main/medical_insurance_cost.csv"
)
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def load_data():
    data = pd.read_csv(DATA_URL)
    print("1. Dataset original")
    print(f"   Filas y columnas: {data.shape}")
    print(data.head(), end="\n\n")
    return data


def clean_data(data):
    clean = data.drop_duplicates().reset_index(drop=True)
    print("2. Limpieza")
    print(f"   Registros eliminados por duplicados: {len(data) - len(clean)}", end="\n\n")
    return clean


def encode_and_scale(data):
    work_data = data.copy()

    # Convertimos texto a números para que el modelo pueda operar con esas variables.
    work_data["sex_n"] = pd.factorize(work_data["sex"])[0]
    work_data["smoker_n"] = pd.factorize(work_data["smoker"])[0]
    work_data["region_n"] = pd.factorize(work_data["region"])[0]

    feature_columns = ["age", "bmi", "children", "sex_n", "smoker_n", "region_n"]
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(work_data[feature_columns])

    scaled_data = pd.DataFrame(scaled_values, columns=feature_columns, index=work_data.index)
    scaled_data["charges"] = work_data["charges"]

    print("3. Codificación y escalado")
    print("   Variables usadas:", feature_columns)
    print(scaled_data.head(), end="\n\n")
    return scaled_data


def select_features(data):
    X = data.drop(columns="charges")
    y = data["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    selector = SelectKBest(score_func=f_regression, k=4)
    selector.fit(X_train, y_train)

    selected_columns = X_train.columns[selector.get_support()].tolist()
    X_train_sel = pd.DataFrame(selector.transform(X_train), columns=selected_columns)
    X_test_sel = pd.DataFrame(selector.transform(X_test), columns=selected_columns)

    print("4. Selección de variables")
    print(f"   Columnas elegidas: {selected_columns}", end="\n\n")
    return X_train_sel, X_test_sel, y_train.reset_index(drop=True), y_test.reset_index(drop=True)


def save_processed_data(X_train, X_test, y_train, y_test):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_data = X_train.copy()
    test_data = X_test.copy()
    train_data["charges"] = y_train
    test_data["charges"] = y_test

    train_path = PROCESSED_DIR / "clean_train.csv"
    test_path = PROCESSED_DIR / "clean_test.csv"
    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    print("5. Archivos guardados")
    print(f"   Train: {train_path}")
    print(f"   Test: {test_path}", end="\n\n")
    return train_data, test_data


def train_and_evaluate(train_data, test_data):
    X_train = train_data.drop(columns="charges")
    y_train = train_data["charges"]
    X_test = test_data.drop(columns="charges")
    y_test = test_data["charges"]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("6. Entrenamiento del modelo")
    print(f"   Intercepto: {model.intercept_:.2f}")
    print(f"   Coeficientes: {model.coef_}", end="\n\n")

    print("7. Evaluación")
    print(f"   MSE: {mean_squared_error(y_test, y_pred):.2f}")
    print(f"   R2: {r2_score(y_test, y_pred):.4f}")


def main():
    data = load_data()
    clean_data_frame = clean_data(data)
    scaled_data = encode_and_scale(clean_data_frame)
    X_train, X_test, y_train, y_test = select_features(scaled_data)
    train_data, test_data = save_processed_data(X_train, X_test, y_train, y_test)
    train_and_evaluate(train_data, test_data)


if __name__ == "__main__":
    main()
