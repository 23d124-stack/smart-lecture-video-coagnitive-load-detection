import torch
import torch.nn as nn
import joblib
import numpy as np

# Define model (same as training)
class CognitiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(30, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        return self.net(x)

# Load model
model = CognitiveModel()
model.load_state_dict(torch.load("cognitive_model.pth"))
model.eval()

# Load scaler
scaler = joblib.load("scaler.pkl")

# Example input (replace with real features)
sample = np.random.randn(30)

# Scale
sample_scaled = scaler.transform(sample.reshape(1, -1))

# Predict
with torch.no_grad():
    logits = model(torch.FloatTensor(sample_scaled))
    probs = torch.softmax(logits, dim=1)

print("Prediction:", probs.argmax().item())
print("Confidence:", probs.max().item())
