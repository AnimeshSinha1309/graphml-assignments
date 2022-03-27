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

def matricize(data):
    print("\\begin{bmatrix}")
    for i in range(len(data)):
        for j in range(len(data[i])):
            print(np.round(data[i, j].item(), 3), '&', end=' ')
        print('\\\\')
    print("\\end{bmatrix}")

print("TARGETS:")

gcn = GCNIllustrated(init=[[1.0, 1.5], [2.0, 1.0]])

# print("INITIAL NODE EMBEDDINGS AND WEIGHTS:")
matricize(gcn.lin.weight.T)
# matricize(gcn[1].lin.weight.T)
# matricize(gcn[2].lin.weight.T)

y = torch.tensor([[1., 0.] for _ in range(len(x))])
optimizer = torch.optim.SGD(gcn.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

gcn.train()

for i in range(3):
    xb = gcn(x, edge_index.T)
    loss = loss_fn(xb, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    matricize(gcn.lin.weight.T.detach().clone())
