import torch
import torch_geometric as pyg

from base import GraphConvolutionalLayer
from engine import training

df = pyg.datasets.Planetoid("datasets/planetoid", "cora")
network = GraphConvolutionalLayer(df.data.x.shape[1], 7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = network.to(device)
df.data = df.data.to(device)

training(network, df.data)
