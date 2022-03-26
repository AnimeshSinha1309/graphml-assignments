import torch
import torch_geometric as pyg

from gcn import GraphConvolutionalNetwork
from engine import training

df = pyg.datasets.Planetoid("datasets/planetoid", "cora")
network = GraphConvolutionalNetwork(df.data.x.shape[1], 10, 7, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = network.to(device)
df.data = df.data.to(device)

print(df.data.edge_index[0])

training(network, df.data)
