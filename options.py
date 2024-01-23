import numpy as np
import torch

class Options:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_epochs = 10
        self.quantiles = np.linspace(0, 1, 101)[1:-1].tolist()
        self.attention_input_size = 1
        self.attention_embed_dim = 100
        self.num_heads = 4
        self.tcn_hidden_size = 10
        self.num_loads = 3
        self.num_weather = 5
        self.num_blocks = len(self.quantiles)
        self.batch = 64
        self.learning_rate = 0.001
        self.train_rate = 0.7
        self.lagged_hour = 3