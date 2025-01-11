import os

from torchviz import make_dot
import torch
import torch.nn as nn

from LoadData.utils import load_config
from model_defination.model_loader import load_model_test

# [Insert model definition here]

# Create the model and generate a dummy input

CONFIG_NAME = "config_train.yaml"
CONFIG_PATH = os.path.join("configs/", CONFIG_NAME)
config = load_config(CONFIG_PATH)

model = load_model_test(config)
dummy_input = torch.randn(1, 3, 256, 256)  # Example input tensor (batch_size=1, channels=3, height=256, width=256)

# Generate the computational graph
dot = make_dot(model(dummy_input), params=dict(model.named_parameters()))

# Save the graph as a file
dot.format = "png"
dot.render("Model_Graph")
