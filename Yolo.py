from ultralytics import YOLO
import ultralytics
import os, shutil
import yaml
import mlflow
import subprocess

# Display an image of the fire detection dataset
from IPython.display import display, Image

# Load and display an image
# Parameters:
#    path (str): Path to the image file
path = r"./datasets/pyronear/images/train/aiformankind_v1_000014.jpg"
img = Image(path, width=500, height=300)
display(img)

# Load model configuration yaml file
# Returns:
#    yolo_params (dict): A dictionary containing YOLO model parameters
with open(r"model_configuration.yaml") as f:
    yolo_params = yaml.safe_load(f)

# Load data configuration yaml file
# Returns:
#    data_params (dict): A dictionary containing data parameters
with open(r"data_configuration.yaml") as f:
    data_params = yaml.safe_load(f)

# Check label names
data_params["names"]

# Training the YOLO Model
# Available models: YOLOV8n, YOLOV8s, YOLOV8m, YOLOV8l, YOLOV8x
# - Build a new model from YAML: `YOLO('yolov8n.yaml')`
# - Load a pre-trained model: `YOLO('yolov8n.pt')`
# - Build from YAML and transfer weights: `YOLO('yolov8n.yaml').load('yolov8n.pt')

# Print chosen YOLO parameters
print("YOLOv8 PARAMETERS:")
print(f"model: {yolo_params['model_type']}")
print(f"imgsz: {yolo_params['imgsz']}")
print(f"lr0: {yolo_params['learning_rate']}")
print(f"batch: {yolo_params['batch']}")
print(f"name: {yolo_params['experiment_name']}")

# Define the YOLO model
model = YOLO(yolo_params['model_type'])

EXPERIMENT_NAME = "pyronear-dev2"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
print("experiment_id:", experiment.experiment_id)

# Functions to get git revision hashes
def get_git_revision_hash():
    """Get the full git revision hash."""
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')

def get_git_revision_short_hash():
    """Get the short git revision hash."""
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')

# Create a directory for the experiment's output
dirpath = os.path.join('./runs/detect/', yolo_params['experiment_name'])
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

# Start a MLflow run
with mlflow.start_run(experiment_id=experiment.experiment_id, run_name="pyronear_yolo") as dl_model_tracking_run:
    tags = {
        "Model": "Yolov8",
        "User": "gbrison",
        "Approach": "best_model"
    }
    mlflow.set_tags(tags)

    # Training the model
    model.train(
        data="data_configuration.yaml",
        imgsz=yolo_params['imgsz'],
        batch=yolo_params['batch'],
        epochs=yolo_params['epochs'],
        optimizer=yolo_params['optimizer'],
        lr0=yolo_params['learning_rate'],
        pretrained=yolo_params['pretrained'],
        name=yolo_params['experiment_name'],
        seed=0
    )

    # Validate the model
    model.val()

    # Log the commit hash
    commit_hash = get_git_revision_hash()
    mlflow.log_param('git_commit_hash', commit_hash)

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
logged_model = f'runs:/{run_id}/model'
model_registry_version = mlflow.register_model(logged_model, 'pyronear_dl_model')
print(f'Model Name: {model_registry_version.name}')
print(f'Model Version: {model_registry_version.version}')

# Command to upload MLflow runs to AWS S3
command = "aws s3 cp --recursive mlruns s3://{bucket-name}/output/mlruns --no-sign-request"
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
print("Output:", stdout.decode('utf-8'))
print("Error:", stderr.decode('utf-8'))
print("Return Code:", process.returncode)
