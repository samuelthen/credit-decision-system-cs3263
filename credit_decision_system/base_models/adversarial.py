import torch
import torch.nn as nn
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# --- 1. Extract sensitive attributes from original dataset ---
def extract_sensitive_attributes(X, original_data):
    original_data_copy = original_data.copy()
    original_data_copy["is_female"] = original_data_copy["personal_status_sex"].isin(["A92", "A95"]).astype(int)
    original_data_copy["is_foreign"] = (original_data_copy["foreign_worker"] == "A201").astype(int)
    return original_data_copy

# --- 2. PyTorch Dataset Wrapper ---
class CreditDataset(Dataset):
    def __init__(self, X, y, sensitive_attr):
        self.X = torch.tensor(X.toarray() if hasattr(X, 'toarray') else X, dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
        self.sensitive = torch.tensor(sensitive_attr.values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.sensitive[idx]

# --- 3. Gradient Reversal Layer (GRL) ---
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

# --- 4. Main Model ---
class MainModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(MainModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.encoder(x)
        output = torch.sigmoid(self.head(features))
        return output, features

# --- 5. Adversary Model ---
class Adversary(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super(Adversary, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        return self.net(features)

# --- 6. Training Loop (with GRL) ---
def train_adversarial(model, adversary, dataloader, num_epochs=20, lambda_grl=1.0):
    model.train()
    adversary.train()

    grl = GradientReversal(lambda_=lambda_grl)
    optimizer_main = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer_adv = torch.optim.Adam(adversary.parameters(), lr=1e-3)
    bce = nn.BCELoss()

    for epoch in range(num_epochs):
        for x, y, sensitive in dataloader:
            # Forward through main model
            y_pred, features = model(x)
            task_loss = bce(y_pred, y)

            # Forward through GRL + adversary
            grl_features = grl(features)
            sensitive_pred = adversary(grl_features)
            adv_loss = bce(sensitive_pred, sensitive)

            # Total loss: joint objective
            total_loss = task_loss + adv_loss

            optimizer_main.zero_grad()
            optimizer_adv.zero_grad()
            total_loss.backward()
            optimizer_main.step()
            optimizer_adv.step()

        print(f"Epoch {epoch+1:02d} | Task Loss: {task_loss.item():.4f} | Adv Loss: {adv_loss.item():.4f}")

__all__ = [
    "extract_sensitive_attributes",
    "CreditDataset",
    "MainModel",
    "Adversary",
    "train_adversarial"
] 
