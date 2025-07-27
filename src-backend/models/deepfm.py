import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepFM(nn.Module):
    def __init__(self, cat_dims, num_dim, embed_dim=8, hidden_dims=[128, 64]):
        super(DeepFM, self).__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, embed_dim) for cat_dim in cat_dims
        ])
        self.linear = nn.Linear(len(cat_dims) * embed_dim + num_dim, 1)

        dnn_input_dim = len(cat_dims) * embed_dim + num_dim
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(dnn_input_dim, h))
            layers.append(nn.ReLU())
            dnn_input_dim = h
        layers.append(nn.Linear(dnn_input_dim, 1))
        self.dnn = nn.Sequential(*layers)

    def fm_layer(self, x):
        sum_square = torch.sum(x, dim=1) ** 2
        square_sum = torch.sum(x ** 2, dim=1)
        return 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)

    def forward(self, cat_feats, num_feats):
        embed = [emb(cat_feats[:, i]) for i, emb in enumerate(self.embeddings)]
        embed = torch.stack(embed, dim=1)
        embed_flat = embed.view(embed.size(0), -1)
        x = torch.cat([embed_flat, num_feats], dim=1)

        linear_out = self.linear(x)
        fm_out = self.fm_layer(embed)
        dnn_out = self.dnn(x)

        return torch.sigmoid(linear_out + fm_out + dnn_out)
