import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import mlflow
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import dvc.api

# Set random seeds for reproducibility
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Define the data paths using DVC
train_data_url = "dvc://data/train-images-idx3-ubyte.gz"
train_labels_url = "dvc://data/train-labels-idx1-ubyte.gz"
test_data_url = "dvc://data/t10k-images-idx3-ubyte.gz"
test_labels_url = "dvc://data/t10k-labels-idx1-ubyte.gz"

# Load MNIST Fashion dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Create train and test datasets
train_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)

# Define the neural network architecture using LightningModule
class Net(LightningModule):
    def __init__(self, learning_rate=0.001):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
        self.learning_rate = learning_rate
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

# Define the hyperparameters
batch_sizes = [128, 256]
num_epochs_list = [5, 10]
learning_rates = [0.001, 0.005]

# Perform hyperparameter search
for batch_size in batch_sizes:
    for num_epochs in num_epochs_list:
        for learning_rate in learning_rates:
            # Create an instance of the neural network model
            model = Net(learning_rate=learning_rate)

            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True
            )

            test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False
            )

            # Define a PyTorch Lightning logger for MLflow
            mlflow_logger = MLFlowLogger(experiment_name="mnist_fashion_experiment")

            # Train the model using PyTorch Lightning
            trainer = Trainer(
                max_epochs=num_epochs,
                logger=mlflow_logger
            )
            trainer.fit(model, train_loader)

            # Evaluate the model on the test set
            model.eval()
            predictions = []
            labels = []

            with torch.no_grad():
                for images, target_labels in test_loader:
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    predictions.extend(predicted.cpu().numpy())
                    labels.extend(target_labels.cpu().numpy())

            # Compute evaluation metrics
            accuracy = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions, average='weighted')
            recall = recall_score(labels, predictions, average='weighted')
            f1 = f1_score(labels, predictions, average='weighted')

            # Log the evaluation metrics to MLflow
            mlflow_logger.log_metrics({
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1
            })

            # Save the model with the hyperparameters as the run name
            mlflow.pytorch.log_model(model, f"model_{batch_size}_{num_epochs}_{learning_rate}")
