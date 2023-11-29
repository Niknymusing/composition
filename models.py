import torch
from torch import nn
from spiralnet import instantiate_model as instantiate_spiralnet 

class SpiralnetClassifierGRU(nn.Module):
    def __init__(self, nr_of_classes, embedding_dim=32, nr_spiralnet_layers=4, nr_rnn_layers=2):
        super(SpiralnetClassifierGRU, self).__init__()
        self.nr_of_gesture_classes = nr_of_classes
        self.embedding_dim = embedding_dim
        self.spiralnet = instantiate_spiralnet(nr_layers=nr_spiralnet_layers, output_dim=self.embedding_dim)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.embedding_dim, nr_rnn_layers, bidirectional=False, batch_first=False)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(self.embedding_dim, self.nr_of_gesture_classes)
        self.softmax = nn.Softmax(dim=1)

        for param in self.gru.parameters():
            if len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)

    def forward(self, x):
        x = self.spiralnet(x)
        x = self.layer_norm(x)
        x, _ = self.gru(x)
        x = self.gelu(x[-1])
        logits = self.fc(x)
        return logits
