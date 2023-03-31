import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import typing as T
import copy
from pathlib import Path


class OpticalFlowCNN(nn.Module):
    def __init__(self, save_path: Path = ".", of_params=None, random_state=42):
        """Initialize the CNN model.

        Parameters:
        ----------
            save_path (Path, optional): Path to save the model. Defaults to ".".
            of_params (T.Optional[T.Dict[str, T.Any]], optional): Optical flow parameters. Defaults to None.
        """

        super(OpticalFlowCNN, self).__init__()

        self.conv1 = nn.Conv2d(10, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 3)

        self.save_path = Path(save_path)
        self.of_params = of_params

    def forward(self, x):
        """Forward pass of the CNN model.

        Parameters:
        ----------
            x (torch.Tensor): Input data.

        Returns:
        -------
            torch.Tensor: Output data.
        """

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def predict(self, X_test):
        """Predict the labels of the test data.

        Parameters:
        ----------
            X_test (np.ndarray): Test data.

        Returns:
        -------
            np.ndarray: Predicted labels.
        """
        self.eval()
        test_dataset = OpticalFlowDataset(X_test)
        return F.softmax(self(test_dataset.X), dim=1).cpu().detach().numpy()

    def train_model(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=32,
        num_epochs=100,
        lr=0.001,
        momentum=0.9,
        patience=10,
        min_delta=0.001,
    ):
        """Train the CNN model.

        Parameters:
        ----------
            X_train (np.ndarray): Training data.
            y_train (np.ndarray): Training labels.
            X_val (np.ndarray): Validation data.
            y_val (np.ndarray): Validation labels.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_epochs (int, optional): Number of epochs. Defaults to 100.
            lr (float, optional): Learning rate. Defaults to 0.001.
            momentum (float, optional): Momentum. Defaults to 0.9.
            patience (int, optional): Patience for early stopping. Defaults to 10.
            min_delta (float, optional): Minimum delta for early stopping. Defaults to 0.001.

        Returns:
        -------
            None
        """

        best_val_loss = float("inf")
        counter = 0

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        optical_flow_data = OpticalFlowDataset(X_train, y_train)

        train_loader = DataLoader(
            optical_flow_data, batch_size=batch_size, shuffle=True
        )

        optical_flow_data_val = OpticalFlowDataset(X_val, y_val)

        val_loader = DataLoader(
            optical_flow_data_val, batch_size=batch_size, shuffle=True
        )

        best_model_state = None

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")

            self.train()

            for i, (inputs, labels) in enumerate(train_loader, 0):
                print(
                    f"Training progress: {int((i+1)/len(train_loader)*100)}%",
                    end="\r",
                )
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Validation loop
            self.eval()

            val_running_loss = 0.0
            for inputs, labels in val_loader:
                print(
                    f"Validation progress: {int((i+1)/len(val_loader)*100)}%",
                    end="\r",
                )
                with torch.no_grad():
                    outputs = self(inputs)
                    val_loss = criterion(outputs, labels)
                    val_running_loss += val_loss.item()

            val_loss = val_running_loss / len(val_loader)

            print(f"Validation loss: {val_loss}")

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                counter = 0
                best_model_state = copy.deepcopy(self.state_dict())
            else:
                counter += 1

            if counter >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break

        self.load_state_dict(best_model_state)

    def predict_all_clips(
        self, all_features: T.Dict[str, np.ndarray]
    ) -> T.Dict[str, np.ndarray]:
        """Predict the labels of all clips.

        Parameters:
        ----------
            all_features (T.Dict[str, np.ndarray]): Dictionary with all features.
                The key is the clip tuple and the value is the features.

        Returns:
        -------
            T.Dict[str, np.ndarray]: Dictionary with all predictions.
                The key is the clip tuple and the value is the predictions.
        """

        self.cpu()
        self.eval()

        # reshape features
        n_layers = self.of_params.n_layers
        grid_size = self.of_params.grid_size

        for key in all_features.keys():

            features_l = all_features[key][:, : (all_features[key].shape[1] // 2)]
            features_r = all_features[key][:, (all_features[key].shape[1] // 2) :]

            features_l = features_l.reshape(-1, n_layers, grid_size, grid_size)
            features_r = features_r.reshape(-1, n_layers, grid_size, grid_size)

            all_features[key] = np.concatenate((features_l, features_r), axis=1)

        return {
            clip_tuple: self.predict(features)
            for clip_tuple, features in all_features.items()
        }

    def save_base_classifier(self, idx) -> None:
        """Save the base classifier.

        Parameters:
        ----------
            idx (int): Index of the base classifier.

        Returns:
        -------
            None
        """
        torch.save(self.state_dict(), self.model_path(idx))

    def model_path(self, idx) -> str:
        return str(self.save_path / f"weights-{idx}.pt")


class OpticalFlowDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
