import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.feature_extraction.text import HashingVectorizer
import json
import logging
from pathlib import Path
from typing import Optional


class ResidualBlock(nn.Module):
    def __init__(self, hidden_size, dropout_prob):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # Skip connection
        return self.relu(out)


class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, num_layers=10, hidden_size=4096, dropout_prob=0.2):
        super(DeepNeuralNetwork, self).__init__()

        # First layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )

        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(num_layers - 2):
            self.residual_blocks.append(ResidualBlock(hidden_size, dropout_prob))

        # Output layer
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.input_layer(x)

        for block in self.residual_blocks:
            x = block(x)

        return self.output_layer(x)


Y_STD = 1.0328539609909058
Y_MEAN = 4.434937953948975


class DeepNeuralNetworkInference:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.device = None

        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    def setup(self):
        self.vectorizer = HashingVectorizer(n_features=5000, stop_words="english", binary=True)
        self.model = DeepNeuralNetwork(5000)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logging.info(f"Neural Network is using {self.device}")

        self.model.to(self.device)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)

    def inference(self, text):
        self.model.eval()
        with torch.no_grad():
            vector = self.vectorizer.transform([text])
            vector = torch.FloatTensor(vector.toarray()).to(self.device)
            pred = self.model(vector)[0]
            result = torch.exp(pred * Y_STD + Y_MEAN) - 1
            result = result.item()
        return max(0, result)


class InrPriceModel:
    """
    The INR model trained by scripts/train_nn_inr.py. Everything needed to rebuild it (layer
    sizes, which input features, the target mean and std of log1p(price) on the train split)
    is read from the meta file saved next to the weights; nothing is hardcoded here.
    """

    def __init__(self, weights: str = "models/nn_inr.pth", meta: Optional[str] = None, device: Optional[str] = None):
        weights_path = Path(weights)
        meta_path = Path(meta) if meta else weights_path.with_name(weights_path.stem + "_meta.json")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.spec = self.meta["features"]
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        from agents.inr_features import input_size

        self.model = DeepNeuralNetwork(
            input_size(self.spec),
            num_layers=self.meta["num_layers"],
            hidden_size=self.meta["hidden_size"],
            dropout_prob=self.meta.get("dropout", 0.2),
        )
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device).eval()
        self.y_mean = float(self.meta["y_mean"])
        self.y_std = float(self.meta["y_std"])

    def predict(self, texts, categories) -> np.ndarray:
        from agents.inr_features import build_features

        features = torch.from_numpy(build_features(texts, categories, self.spec)).to(self.device)
        with torch.no_grad():
            out = self.model(features).squeeze(1).float().cpu().numpy()
        return np.maximum(np.expm1(out * self.y_std + self.y_mean), 0.0)

    def price(self, text: str, app_category: Optional[str] = None) -> float:
        """Price one item. The taxonomy feature comes from the title rules, then the app category."""
        from agents.taxonomy import taxonomy_for

        return float(self.predict([text], [taxonomy_for(text, app_category)])[0])
