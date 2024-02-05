

# Pyronear - Machine Learning Pipeline for Wildfire Detection 🚀

## Train YOLO Wildfire Detector Using Ultralytics

This repository showcases how to train a YOLOv8 deep learning model on the Pyronear dataset. Key features include the use of DVC for data versioning and MLflow for model versioning and performance tracking, with cloud storage for data.

![ML flow](https://i.postimg.cc/mrsjc7yY/mlflow.png)

### Prerequisites
- Python 3.x
- Pip package manager

### 01. Install Dependencies 📦

To install necessary libraries, run:

```shell
pip install -r requirements.txt
```

### 02. Download and Prepare Dataset 📥

1. Download the dataset using the following command:

    ```shell
    gdown --fuzzy https://drive.google.com/file/d/12gGuFd3aQmtPXP-cbBRjsciWLtpFNBB-/view?usp=sharing
    ```

2. Unzip and organize the dataset:

    ```shell
    mkdir datasets
    unzip DS-18d12de1.zip -d datasets/
    ```

3. Update the dataset path in `data_configuration.yaml`.

### 03. Data Overview 💽

The dataset comprises 596 training images and 148 validation images featuring forest landscapes with smoke. Each image (640x480 pixels) is annotated with a bounding box in a corresponding txt file, marking the smoke areas.

![Fumes](https://i.postimg.cc/sxPwsrxR/aiformankind-v1-000007.jpg)

### 04. Data Version Control (DVC) 🔄

#### 1️⃣ Install DVC:

Use the same requirements file to install DVC.

#### 2️⃣ Data Setup:

- Initialize DVC in your workspace:

    ```sh
    dvc init
    ```

- Set up remote storage (e.g., AWS S3, Google Cloud Storage):

    ```sh
    dvc remote add -d remote_storage path/to/your/dvc_remote
    ```

- Track data and configuration files using DVC:

    ```sh
    dvc add <file_or_directory>
    git add .dvc/<file_or_directory>.dvc .gitignore
    ```

### 05. Model Training with YOLOv8 and MLflow Tracking 🤖

⚠️ Requires GPU.

#### 3️⃣ MLflow Setup:

MLflow is used for experiment tracking and model management. Key tracked metrics include epochs, accuracy, and loss.

- Start the MLflow UI:

    ```sh
    mlflow ui
    ```

- (Optional) Specify a custom port:

    ```sh
    mlflow ui --port <port_number>
    ```

#### 4️⃣ Training the Model:

Execute the training script with specified data and model configurations:

```shell
python3 train_yolo.py --data_config data_configuration.yaml --model_config model_configuration.yaml
```

#### 5️⃣ Cloud Storage for Artifacts:

Add AWS credentials to the training script:

```sh
"s3", aws_access_key_id="your_access_key_id", aws_secret_access_key="your_secret_access_key"
```

### Congratulations! 🎉

You've successfully set up and run the Pyronear machine learning pipeline for wildfire detection.
