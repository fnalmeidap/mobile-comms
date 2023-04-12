import torch
import torch.backends.mps

from torch import nn
from mobcom.environment import ModelNN

class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(ModelNN.L1_INPUT_SIZE, ModelNN.L1_OUTPUT_SIZE),
            nn.ReLU(),
            nn.Linear(ModelNN.L2_INPUT_SIZE, ModelNN.L2_OUTPUT_SIZE),
            nn.ReLU(),
            nn.Linear(ModelNN.L3_INPUT_SIZE, ModelNN.L3_OUTPUT_SIZE)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)
        
