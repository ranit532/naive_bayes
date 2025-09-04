import sys
import codecs
import os
import pandas as pd
import mlflow
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.datasets import load_iris

# Redirect stdout to a file with utf-8 encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach(), 'strict')

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Naive Bayes Classifiers")

def plot_confusion_matrix(cm, model_name, file_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(file_path)
    plt.close()

def train_gaussian_nb():
    with mlflow.start_run(run_name="GaussianNB_Iris") as run:
        # Load data
        iris = load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Log parameters
        mlflow.log_param("model_type", "GaussianNB")

        # Log metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Generate and log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plot_path = f"plots/confusion_matrix_GaussianNB_{int(time.time())}.png"
        plot_confusion_matrix(cm, "GaussianNB", plot_path)
        mlflow.log_artifact(plot_path)

        # Save and log model
        model_path = f"models/gaussian_nb_{int(time.time())}.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        print(f"GaussianNB model trained and logged. Run ID: {run.info.run_id}")

def train_multinomial_nb():
    with mlflow.start_run(run_name="MultinomialNB_SyntheticText") as run:
        # Load data
        df = pd.read_csv("data/synthetic_text_data.csv")
        X = df['text']
        y = df['category']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Vectorize text
        vectorizer = CountVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Train model
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        # Log parameters
        mlflow.log_param("model_type", "MultinomialNB")

        # Log metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Generate and log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plot_path = f"plots/confusion_matrix_MultinomialNB_{int(time.time())}.png"
        plot_confusion_matrix(cm, "MultinomialNB", plot_path)
        mlflow.log_artifact(plot_path)

        # Save and log model
        model_path = f"models/multinomial_nb_{int(time.time())}.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        
        # Save vectorizer
        vectorizer_path = f"models/multinomial_nb_vectorizer_{int(time.time())}.joblib"
        joblib.dump(vectorizer, vectorizer_path)
        mlflow.log_artifact(vectorizer_path)


        print(f"MultinomialNB model trained and logged. Run ID: {run.info.run_id}")

def train_bernoulli_nb():
    with mlflow.start_run(run_name="BernoulliNB_SyntheticText") as run:
        # Load data
        df = pd.read_csv("data/synthetic_text_data.csv")
        X = df['text']
        y = df['category']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Vectorize text (binary)
        vectorizer = CountVectorizer(binary=True)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        # Train model
        model = BernoulliNB()
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        # Log parameters
        mlflow.log_param("model_type", "BernoulliNB")

        # Log metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Generate and log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plot_path = f"plots/confusion_matrix_BernoulliNB_{int(time.time())}.png"
        plot_confusion_matrix(cm, "BernoulliNB", plot_path)
        mlflow.log_artifact(plot_path)

        # Save and log model
        model_path = f"models/bernoulli_nb_{int(time.time())}.joblib"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        
        # Save vectorizer
        vectorizer_path = f"models/bernoulli_nb_vectorizer_{int(time.time())}.joblib"
        joblib.dump(vectorizer, vectorizer_path)
        mlflow.log_artifact(vectorizer_path)

        print(f"BernoulliNB model trained and logged. Run ID: {run.info.run_id}")

if __name__ == "__main__":
    # Create directories if they don't exist
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists("plots"):
        os.makedirs("plots")

    train_gaussian_nb()
    train_multinomial_nb()
    train_bernoulli_nb()
