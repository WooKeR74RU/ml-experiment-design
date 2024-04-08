import boto3
import mlflow.sklearn
import os
import pandas as pd

from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
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

    # Векторизация
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Обучение модели
    clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    clf.fit(X_train_vec, y_train)

    # Предсказание
    y_pred = clf.predict(X_test_vec)

    os.environ["MLFLOW_TRACKING_URI"] = f"http://host.docker.internal:5000"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = f"http://host.docker.internal:9000"

    # Настройка клиента boto3
    boto3.setup_default_session(
        aws_access_key_id="some_aws_access_key_id",
        aws_secret_access_key="some_aws_secret_access_key",
        region_name="us-west-1",
    )

    # Логирование в MLflow
    with mlflow.start_run() as run:
        # Логирование параметров и метрик
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))

        # Логирование модели
        mlflow.sklearn.log_model(clf, "model", registered_model_name="SomeModel")

    print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")
