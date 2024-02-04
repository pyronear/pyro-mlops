"""
Authors: Matéo Lostanlen & Gaëtan Brison
Date: 22 Dec 2024
Goal: Yolov8 model trained on Pyronear Dataset with parameters of configuration
 saved thanks to mlflow. This code outputs the artifacts of the top model in aws.
"""

# 01 - Import Libraries

import argparse
import os
import shutil
import subprocess

import boto3
import mlflow
import pandas as pd
import yaml

# Display an image of the fire detection dataset
from IPython.display import Image, display
from ultralytics import YOLO

# 02 Define methods used later on

# Functions to get git revision hashes


def get_git_revision_hash():
    """Get the full git revision hash."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")


def get_git_revision_short_hash():
    """Get the short git revision hash."""
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode("utf-8")


# Selection of the top model and uploading the artefact on AWS


def find_and_concatenate_csvs(root_dir, output_file):
    """
    Searches for CSV files named 'results.csv' in a directory tree and concatenates them into a single CSV file.

    This function traverses through all subdirectories of a given root directory, looking for files named 'results.csv'.
    Each found CSV file is read into a DataFrame, a column indicating its path is added, and then all are concatenated
    into a single DataFrame. This concatenated DataFrame is saved to a specified output file and returned.

    Parameters
    ----------
    root_dir : str
        The root directory to start the search for CSV files.
    output_file : str
        The path of the output file where the concatenated CSV will be saved.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the concatenated contents of all found CSV files.
    """

    all_dfs = []  # List to store all dataframes
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == "results.csv":
                path = os.path.join(root, file)
                print(path)
                df = pd.read_csv(path)
                df["Path"] = path  # Add a column with the path
                all_dfs.append(df)

    # Concatenate all dataframes
    concatenated_df = pd.concat(all_dfs, ignore_index=True)
    concatenated_df.to_csv(output_file, index=False)
    print(f"Concatenated CSV saved as {output_file}")
    return concatenated_df


def upload_directory_to_s3(bucket_name, directory_name, s3_folder):
    """
    Uploads a local directory to an S3 bucket.

    This function traverses through all files within a specified local directory (including its subdirectories),
    and uploads each file to an AWS S3 bucket. The files are stored in the specified S3 folder within the bucket.
    The AWS credentials are hardcoded within the function.

    Parameters
    ----------
    bucket_name : str
        The name of the AWS S3 bucket where the directory is to be uploaded.
    directory_name : str
        The name of the local directory to upload.
    s3_folder : str
        The name of the folder within the S3 bucket where files from the directory will be stored.

    Returns
    -------
    None
    """

    s3_client = boto3.client(
        "s3", aws_access_key_id="your_access_key_id", aws_secret_access_key="your_secret_access_key"
    )
    for subdir, dirs, files in os.walk(directory_name):
        for file in files:
            full_path = os.path.join(subdir, file)
            with open(full_path, "rb") as data:
                s3_client.upload_fileobj(data, bucket_name, s3_folder + "/" + full_path[len(directory_name) + 1 :])
    return print("mlruns uploaded on aws")


# 03 Input Configuration

# Create the argument parser
parser = argparse.ArgumentParser(description="Load YAML configuration files for YOLO model and data")

# Add arguments for the paths to the YAML files
parser.add_argument("--data_config", type=str, required=True, help="Path to the data configuration YAML file")
parser.add_argument("--model_config", type=str, required=True, help="Path to the model configuration YAML file")


# Parse the arguments
args = parser.parse_args()

# Load the data configuration yaml file
with open(args.data_config) as f:
    data_params = yaml.safe_load(f)
print("Data Parameters:", data_params)

# Check label names
data_params["names"]

# Load the model configuration yaml file
with open(args.model_config) as f:
    yolo_params = yaml.safe_load(f)
print("Model Parameters:", yolo_params)


# 04 Training the YOLO Model

# Print chosen YOLO parameters
print("YOLOv8 PARAMETERS:")
print(f"""model: {yolo_params['model_type']}""")
print(f"imgsz: {yolo_params['imgsz']}")
print(f"lr0: {yolo_params['learning_rate']}")
print(f"batch: {yolo_params['batch']}")
print(f"name: {yolo_params['experiment_name']}")
print(yolo_params)

# Define the YOLO model
model = YOLO(yolo_params["model_type"])

EXPERIMENT_NAME = "pyronear-dev2"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
print("experiment_id:", experiment.experiment_id)

# Create a directory for the experiment's output
dirpath = os.path.join("./runs/detect/", yolo_params["experiment_name"])
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

# Start a MLflow run
with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="pyronear_yolo") as dl_model_tracking_run:
    tags = {"Model": "Yolov8", "User": "gbrison", "Approach": "best_model"}
    mlflow.set_tags(tags)

    # Training the model
    model.train(
        data="data_configuration.yaml",
        imgsz=yolo_params["imgsz"],
        batch=yolo_params["batch"],
        epochs=yolo_params["epochs"],
        optimizer=yolo_params["optimizer"],
        lr0=yolo_params["learning_rate"],
        pretrained=yolo_params["pretrained"],
        name=yolo_params["experiment_name"],
        seed=0,
    )

    # Validate the model
    model.val()

    # Log the commit hash
    commit_hash = get_git_revision_hash()
    mlflow.log_param("git_commit_hash", commit_hash)

    # Track dependencies
    installed_packages = subprocess.check_output(["pip", "freeze"]).decode("utf-8")
    mlflow.log_param("dependencies", installed_packages)

# Display confusion matrix and validation images
path = f"./runs/detect/{yolo_params['experiment_name']}"
confusion_matrix = Image(os.path.join(path, "confusion_matrix_normalized.png"), width=800, height=600)
display(confusion_matrix)

val_images = Image(os.path.join(path, "val_batch0_labels.jpg"), width=800, height=600)
display(val_images)

# Retrieve and print run information
run_id = dl_model_tracking_run.info.run_id
print("run_id: {}; lifecycle_stage: {}".format(run_id, mlflow.get_run(run_id).info.lifecycle_stage))

# Register the model
logged_model = f"runs:/{run_id}/model"
model_registry_version = mlflow.register_model(logged_model, "pyronear_dl_model")
print(f"Model Name: {model_registry_version.name}")
print(f"Model Version: {model_registry_version.version}")

# 05 Store runs configurations and performance in csv

output_csv_file = "concatenated_results.csv"  # Output file name
df = find_and_concatenate_csvs("../../mlartifacts", output_csv_file)
df.columns = df.columns.str.replace(" ", "")
df = df.reset_index(drop=True)
sorted_df = df.sort_values("metrics/mAP50-95(B)", ascending=False)
print(sorted_df.head())
print("Top metrics/mAP50-95(B):  ")
print("    ")
print("    ")
print(list(sorted_df["metrics/mAP50-95(B)"][0:1])[0])
print("    ")
print("    ")
print("Obtained with artefact: ", list(sorted_df["Path"][0:1])[0])

final_model = list(sorted_df["Path"][0:1])[0]

# 06 Upload models to AWS
bucket_name = "pyronear-v2"
local_directory = "../../mlruns"
s3_folder = "output/mlruns"

upload_directory_to_s3(bucket_name, local_directory, s3_folder)
