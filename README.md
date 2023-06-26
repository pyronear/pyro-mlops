# Pytorch + DVC + MLFlow - Documentation 🚀

## Train yolo wildfire detector using ultralytics

### Install dependencies

```shell
pip install -r model/yolo/requirements.txt
```

### Download dataset

```shell
gdown --fuzzy https://drive.google.com/file/d/12gGuFd3aQmtPXP-cbBRjsciWLtpFNBB-/view?usp=sharing
```

Unzip dataset

```shell
unzip DS-18d12de1.zip
```

update dataset path in DS-18d12de1/data.yaml

### Train model

```shell
yolo train cfg=model/yolo/yolo_config.yaml  data=DS-18d12de1/data.yaml model=yolov8n.pt  
```


## FashionMNIST Classification with PyTorch

This repository contains an example of training a deep learning model on the FashionMNIST dataset using PyTorch. It demonstrates how to use DVC for tracking the Data versions and MLflow for tracking the model versions and performances.

![Pipeline](https://i.postimg.cc/Y2WZj4qQ/Screenshot-2023-06-07-at-17-20-47.png)


## 💽 Data used

The dataset consists of 70,000 grayscale images, with each image being a 28x28 pixel representation. These images are divided into 60,000 training examples and 10,000 testing examples.

The Fashion MNIST dataset consists of ten categories or classes. Each class represents a different type of clothing item. The categories are as follows:
T-shirt / Trouser / Pullover / Dress / Coat / Sandal / Shirt / Sneaker / Bag / Ankle / boot

These categories cover a diverse range of fashion items, providing a comprehensive dataset for training and evaluating machine learning models in the domain of fashion image classification.

## 🤖 Pytorch Model used 

The provided code snippet represents a neural network called "FashionClassifier." It is designed for classifying fashion images into 10 different categories. Let's break down its structure and operations:

Architecture:

- Input Layer: The network expects grayscale images as input with a single channel (1).
Convolutional Layer 1: It applies a 2D convolution operation with 32 filters, each having a kernel size of 3x3. The stride is set to 1, and the padding is 1 to maintain the spatial dimensions of the input.
- ReLU Activation 1: A rectified linear unit (ReLU) activation function is applied element-wise to introduce non-linearity after the first convolutional layer.
- Max Pooling 1: It performs 2x2 max pooling, reducing the spatial dimensions of the input by a factor of 2.
- Convolutional Layer 2: Similar to the first convolutional layer, this layer applies a 2D convolution with 64 filters, kernel size of 3x3, stride of 1, and padding of 1.
- ReLU Activation 2: Another ReLU activation is applied after the second convolutional layer.
- Max Pooling 2: Another 2x2 max pooling operation is performed, further reducing the spatial dimensions.
- Fully Connected Layer (Output Layer): It reshapes the 2D feature maps into a flat vector by using the view function, making it compatible for feeding into a fully connected layer. The size of this layer is determined by multiplying the dimensions of the feature maps (7x7x64). The output size of this layer is 10, corresponding to the 10 fashion categories to be classified.


## ➡️ Steps to run the model using DVC and ML Flow: 

#### 1️⃣ Install the required packages:
- pip install -r requirements.txt

#### 2️⃣ Data Setup
- Initialize DVC and set up remote storage (if necessary):
 ```sh
dvc init
```
Using dvc init in workspace will initialize a DVC project, including the internal .dvc/ directory

- Configure DVC remote storage (e.g., AWS S3, Google Cloud Storage):
this will add data to remote storage
 ```sh
dvc remote add -d remote_storage path/to/your/dvc_remote
```
dvc add copies the specified directory or files to .dvc/cache or shared_cache/you/specified, creates .dvc files for each tracked folder or file and adds them to .gitignore
* .dvc and other files are tracked with git add --all



#### 3️⃣ MLflow
MLflow helps in tracking experiments, packaging code into reproducible runs, and sharing and deploying models. You can find more information about MLflow. We have used MLflow to track the experiments and save parameters and metrics used for a particular training. We can include or change parameters according to our requirements
**Tracked Parameters and Metrics:**
- Epochs
- Accuracy
- Loss

##### 3.1 Access the MLflow UI in your browser after running you script:
Run  the below command
```sh
    mlflow ui
```
It will host you on the local computer. compare model seeing metrics


##### 3️.2 Unmanaged without MLflow CLI
Run the standard main function from the command-line in the model path
 ```sh
python model_pytorch_lightning_mnist.py 
```

##### 3️.3 MLflow CLI - mlflow run
Use the MLproject file. We get more control over an MLflow Project by adding an MLproject file, which is a text file in YAML syntax, to the project’s root directory.

- mlflow run local
 ```sh
mlflow run "model_pytorch_lightning_mnist" -P n_epochs=5
```

- mlflow run github
 ```sh
mlflow run https://github.com/<username>/<filename>.git -P <parameter1>=<value>
```

### Congrats you made it 🎉
 
