
import copy
from os import path
from typing import Dict

import torch
from torch import nn
from torch.utils.data import TensorDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm


def transform_cifar10():
    "transforms for the cifar 10"
    return Compose(
        [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    )

def sample_from_class(data_set, k):
    """
    function to sample data and their labels from a dataset in pytorch in
    a stratified manner
    Args
    ----
    data_set
    k: the number of samples that will be accuimulated in the new slit
    Returns
    -----
    train_dataset
    val_dataset
    """
    class_counts = {}
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    for data, label in data_set:
        class_i = label.item() if isinstance(label, torch.Tensor) else label
        class_counts[class_i] = class_counts.get(class_i, 0) + 1
        if class_counts[class_i] <= k:
            train_data.append(data)
            train_label.append(label)
        else:
            test_data.append(data)
            test_label.append(label)

    train_data = torch.stack(train_data)
    train_label = torch.tensor(train_label, dtype=torch.int64)
    test_data = torch.stack(test_data)
    test_label = torch.tensor(test_label, dtype=torch.int64)

    return (
        TensorDataset(train_data, train_label),
        TensorDataset(test_data, test_label),
    )


def load_cifar():
    train_set = CIFAR10(
        root="./data/data_cifar10/",
        train=True,
        transform=transform_cifar10(),
        download=True
        )
    val_set, tr_set = sample_from_class(train_set, 500)
    test_set = CIFAR10(
        root="./data/data_cifar10/",
        train=False,
        transform=transform_cifar10(),
        download=True
        )
    return [train_set, val_set, test_set]

class SampleCNN(nn.Module):
    def __init__(self, shape=(3, 32, 32), batch_size=4):
        super().__init__()
        self.input_shape = shape
        self.batch_size = batch_size

        self.conv1 = nn.Conv2d(in_channels=shape[0], out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3)
        self.pool2 = nn.AvgPool2d(2)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3)
        self.pool3 = nn.AvgPool2d(2)
        self.relu3 = nn.ReLU()

        self.flatten = nn.Flatten()
        self.interface_shape = self.get_shape()
        self.interface = nn.Linear(in_features=self.interface_shape.numel(), out_features=10)

    def get_shape(self):
        sample = torch.randn(size=(self.batch_size, *self.input_shape))
        out = self.conv1(sample)
        out = self.pool1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.pool3(out)
        out = self.relu3(out)
        return out.shape[1:]

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.pool3(out)
        out = self.relu3(out)
        out = self.flatten(out)
        return self.interface(out)


class SimpleTrainer:
    def __init__(
        self, datasets=None, dataloaders=None, models_path=".", cuda="cuda:0",
    ):
        super().__init__()
        self.datasets = datasets
        # TODO: choose GPU with less memory
        self.devicy = torch.device(
            cuda if torch.cuda.is_available() else "cpu"
        )
        self.datasizes = {
            i: len(sett)
            for i, sett in zip(["train", "val", "test"], self.datasets)
        }
        self.models_path = models_path
        self.dataloaders = dataloaders
        self.criterion = nn.CrossEntropyLoss()

    def train(
        self, net: nn.Module, parameters: Dict[str, float], name: str,
    ) -> nn.Module:

        net.to(device=self.devicy)  # pyre-ignore [28]
        optimizer = torch.optim.Adam(
            net.parameters(), lr=parameters.get("learning_rate")
        )
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=parameters.get("learning_step"),
            gamma=parameters.get("learning_gamma"),
        )

        # Train Network
        net = self.train_loop(
            net, optimizer, exp_lr_scheduler, name, parameters.get("epochs")
        )
        return net

    def train_loop(self, model, optimizer, scheduler, name, epochs):
        """
        Training loop
        """
        best_loss = 10 ** 8
        t = tqdm(range(epochs))
        for _ in t:
            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0.0

                for inputs, labels in self.dataloaders[phase]:

                    inputs = inputs.to(self.devicy)
                    labels = labels.to(self.devicy)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * labels.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    if phase == "train":
                        scheduler.step()

                epoch_acc = running_corrects / self.datasizes[phase]
                epoch_loss = running_loss / self.datasizes[phase]
                # deep copy the model
                if phase == "val" and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    t.set_description("Validation accuracy %.3f" % epoch_acc)
                    t.refresh()
        model.load_state_dict(best_model_wts)
        torch.save(
            model.state_dict(), path.join(self.models_path, str(name) + ".pth")
        )
        return model

    def evaluate(self, net: nn.Module) -> float:

        correct = 0
        total = 0
        data_loader = self.dataloaders["test"]
        net.eval()
        with torch.no_grad():

            for inputs, labels in data_loader:
                inputs = inputs.to(device=self.devicy)
                labels = labels.to(device=self.devicy)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total
