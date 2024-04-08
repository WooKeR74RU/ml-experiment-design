import pip

if __name__ == "__main__":
    pip.main(["install", "hyperopt"])  # TODO: fix it

import boto3
import mlflow.sklearn
import os
import pandas as pd

from hyperopt import hp, fmin, tpe
from io import StringIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    pip.main(["install", "hyperopt"])  # TODO: fix it

    # Инициализация клиента
    s3 = boto3.client(
        "s3",
        endpoint_url=f"http://host.docker.internal:9000",
        aws_access_key_id="some_aws_access_key_id",
        aws_secret_access_key="some_aws_secret_access_key",
    )

    # Считывание данных
    obj = s3.get_object(Bucket="datasets", Key="kinopoisk_train.csv")
    data = obj["Body"].read().decode("utf-8")
    df = pd.read_csv(StringIO(data))

    X_train, X_test, y_train, y_test = train_test_split(
        df["review"], df["sentiment"], test_size=0.2, random_state=42
    )

    def objective(params):
        model_type, vectorizer_type = params["model"], params["vectorizer"]

        if model_type == "LogisticRegression":
            clf = LogisticRegression(C=params["C"], random_state=42)
        elif model_type == "RandomForestClassifier":
            clf = RandomForestClassifier(
                n_estimators=params["n_estimators"], random_state=42
            )
        else:
            raise NotImplementedError("Not implemented model")

        if vectorizer_type == "CountVectorizer":
            vectorizer = CountVectorizer()
        elif vectorizer_type == "TfidfVectorizer":
            vectorizer = TfidfVectorizer()
        else:
            raise NotImplementedError("Not implemented vectorizer")

        # Векторизация
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Обучение модели
        clf.fit(X_train_vec, y_train)

        # Предсказание
        y_pred = clf.predict(X_test_vec)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        print(f"Params: {params}")

        # Логирование в MLflow
        with mlflow.start_run() as run:
            # Логирование параметров и метрик
            mlflow.log_param("model_type", model_type)
            mlflow.log_metric("accuracy", accuracy)

            # Логирование модели
            mlflow.sklearn.log_model(clf, "model", registered_model_name="SomeModel")

        return -accuracy

    os.environ["MLFLOW_TRACKING_URI"] = f"http://host.docker.internal:5000"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://host.docker.internal:9000"

    # Настройка клиента boto3
    boto3.setup_default_session(
        aws_access_key_id="some_aws_access_key_id",
        aws_secret_access_key="some_aws_secret_access_key",
        region_name="us-west-1",
    )

    logistic_regression_space = {
        "model": "LogisticRegression",
        "vectorizer": hp.choice("vectorizer", ["CountVectorizer", "TfidfVectorizer"]),
        "C": hp.uniform("C", 10**-2, 10**2),
    }
    fmin(fn=objective, space=logistic_regression_space, algo=tpe.suggest, max_evals=10)

    random_forest_classifier_space = {
        "model": "RandomForestClassifier",
        "vectorizer": hp.choice("vectorizer", ["CountVectorizer", "TfidfVectorizer"]),
        "n_estimators": hp.uniformint("n_estimators", 10, 100),
    }
    fmin(
        fn=objective,
        space=random_forest_classifier_space,
        algo=tpe.suggest,
        max_evals=10,
    )
