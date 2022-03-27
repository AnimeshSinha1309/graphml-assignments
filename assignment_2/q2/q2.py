import numpy as np
import torch
import torch_geometric as pyg

torch.random.manual_seed(0)
np.random.seed(0)

class GCNIllustrated(pyg.nn.GCNConv):

    def __init__(self, in_channels=2, out_channels=2, init=[[1, 1.5], [2.0, 1.0]]) -> None:
        super().__init__(in_channels, out_channels, bias=False)
        self.lin.weight = torch.nn.Parameter(torch.tensor(init))


x = torch.ones((10, 2))
edge_index = torch.tensor([
    [0, 1],
    [0, 2],
    [0, 3],
    [0, 4],
    [1, 5],
    [1, 6],
    [1, 7],
    [7, 6],
    [3, 8],
    [6, 9],
])
targets = torch.randint(0, 3, (10, 1))

def matricize(data):
    print("\\begin{bmatrix}")
    for i in range(len(data)):
        for j in range(len(data[i])):
            print(np.round(data[i, j].item(), 3), '&', end=' ')
        print('\\\\')
    print("\\end{bmatrix}")

print("TARGETS:")
matricize(targets)

gcn = torch.nn.ModuleList([
    GCNIllustrated(init=[[1.0, 1.5], [2.0, 1.0]]),
    GCNIllustrated(init=[[0.3, 1.9], [0.6, 2.2]]),
    GCNIllustrated(init=[[0.8, 1.2], [2.3, 0.9]]),
])

print("INITIAL NODE EMBEDDINGS AND WEIGHTS:")
matricize(x.T)
for i in range(1):
    x = gcn[0](x, edge_index.T)
    print(f"ITERATION {i} Layer 1 NODE EMBEDDINGS AND WEIGHTS:")
    matricize(x.T)
    x = gcn[1](x, edge_index.T)
    print(f"ITERATION {i} Layer 2 NODE EMBEDDINGS AND WEIGHTS:")
    matricize(x.T)
    x = gcn[2](x, edge_index.T)
    print(f"ITERATION {i} Layer 3 NODE EMBEDDINGS AND WEIGHTS:")
    matricize(x.T)
