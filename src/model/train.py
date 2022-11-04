# Import libraries

import argparse
import glob
import os
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from mlflow.tracking import MlflowClient

# define functions
def main(args):
    # TO DO: enable autologging
    mlflow.autolog()
    # read data
    df = get_csvs_df(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    train_model(
        args.reg_rate, "", args.run_id, X_train, X_test, y_train, y_test
    )


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


# TO DO: add function to split data
def split_data(df):
    X, y = (
        df[
            [
                "Pregnancies",
                "PlasmaGlucose",
                "DiastolicBloodPressure",
                "TricepsThickness",
                "SerumInsulin",
                "BMI",
                "DiabetesPedigree",
                "Age",
            ]
        ].values,
        df["Diabetic"].values,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=0
    )
    return X_train, X_test, y_train, y_test


def train_model(
    reg_rate, environment, run_id, X_train, X_test, y_train, y_test
):
    # train model
    with mlflow.start_run() as run:
        LogisticRegression(C=1 / reg_rate, solver="liblinear").fit(
            X_train, y_train
        )
    model_name = "diabetes-production-model"
    print("Registering the model via MLFlow")
    print(model_name)
    # mlflow.sklearn.log_model(
    #     sk_model=trained_model,
    #     registered_model_name=model_name,
    #     artifact_path=model_name,
    # )

    model_path = "model"
    model_uri = "runs:/{}/{}".format(run.info.run_id, model_path)
    mlflow.register_model(model_uri, model_name)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest="training_data", type=str)
    parser.add_argument(
        "--reg_rate", dest="reg_rate", type=float, default=0.01
    )
    parser.add_argument("--run_id", dest="run_id", type=str)
    parser.add_argument("--environment", dest="environment", type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
