#1
import os
import sys
import time
import logging
from urllib.parse import urlparse

#2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from mlflow.models.signature import infer_signature
import mlflow.pytorch
from mlflow import MlflowClient
import mlflow



#enable debug logging, and the full traceback will be displayed, including the detailed error message
logging.basicConfig(level=logging.WARN)
#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Define the transformation to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST Fashion dataset
trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Define the data loaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)



# Define the model 
class FashionClassifier(nn.Module):
    def __init__(self):
        super(FashionClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(7 * 7 * 64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(-1, 7 * 7 * 64)
        x = self.fc(x)
        return x

def log_scalar(name, value, step):
    """Log a scalar value to both MLflow and TensorBoard"""
    mlflow.log_metric(name, value, step=step)


#mlflow start

# defining a new experiment
experiment_name = 'ML_TORCH1'

# returns experiment ID
tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
try:
    # creating a new experiment
    exp_id = mlflow.create_experiment(name=experiment_name)
except Exception as e:
    exp_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

#CONFIRM THIS
#signature = infer_signature(trainset, testset)
#so as to get different pytorch model
timestamp = int(time.time()) 
#model_dir = f"/mlflow_torch/models/{timestamp}"
model_dir = f"ml_torch/models/{timestamp}"

data_dir = "data/FashionMNIST/raw"
if __name__ == "__main__":

    # Number of epochs to train for 
    #   n_epochs = 10
    n_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 5


    # log the model
    with mlflow.start_run(experiment_id=exp_id, run_name = 'first_pytorch_run') as run:
        # adding tags to the run
        mlflow.set_tag('Description','Simple MNIST pytorch Model')
        mlflow.set_tags({'ProblemType': 'Classification', 'ModelLibrary': 'pytorch'})


        # Create an instance of the model
        model = FashionClassifier()

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        # Train the model
        for epoch in range(n_epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 200 == 199:
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
                    running_loss = 0.0
                
                log_scalar('loss', running_loss/ 200, epoch)
        print('Finished training')

        # Test the model
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print('Accuracy on the test set: %.2f %%' % accuracy)

        if os.path.exists(data_dir):
            mlflow.log_artifact(data_dir)
        # logging parameters 
        mlflow.log_param("epochs", n_epochs)

        # logging metrics
        mlflow.log_metric("accuracy", accuracy)
        
        #model name
        mlflow.pytorch.log_model(model, "models")

        #mlflow.pytorch.log_model(model, "models",signature=signature)
        mlflow.pytorch.save_model(model, path = model_dir)
        
        #mlflow.pytorch.save_model(model, pytorch_model_path)
        mlflow.end_run()