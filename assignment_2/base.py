import torch

from engine import training


class GraphConvolutionalLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.w = torch.nn.Linear(
            in_channels,
            out_channels,
            bias=False,
        )
        self.b = torch.nn.Linear(
            in_channels,
            out_channels,
            bias=False,
        )
        self.f = torch.nn.ReLU()

    def forward(self, x, edge_index):
        aggregated_neighbors = [[] for _ in range(x.shape[0])]
        for edge in edge_index:
            i, j = edge[0], edge[1]
            aggregated_neighbors[i].append(x[j])
            aggregated_neighbors[j].append(x[i])

        final_outputs = []
        for i in range(len(x)):
            aggregated_neighbor_tensor = (
                torch.mean(torch.stack(aggregated_neighbors[i], dim=0), dim=0)
                if len(aggregated_neighbors[i]) > 0
                else torch.zeros(self.in_channels)
            )
            final_outputs.append(
                self.f(self.w(aggregated_neighbor_tensor) + self.b(x[i]))
            )
        final_outputs = torch.stack(final_outputs, dim=0)

        return final_outputs
