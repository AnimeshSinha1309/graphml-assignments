import abc

import torch

from engine import training


class GraphNeuralNetwork(torch.nn.Module, abc.ABC):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index):
        aggregated_neighbors = [[] for _ in range(x.shape[0])]
        for edge in edge_index:
            i, j = edge[0], edge[1]
            aggregated_neighbors[i].append(x[j])
            aggregated_neighbors[j].append(x[i])

        final_outputs = []
        for i in range(len(x)):
            aggregated_neighbor_tensor = self.aggregate(aggregated_neighbors[i])
            final_outputs.append(self.combine(aggregated_neighbor_tensor, x[i]))
        final_outputs = torch.stack(final_outputs, dim=0)
        return final_outputs

    @abc.abstractmethod
    def aggregate(self, aggregated_neighbors):
        raise NotImplementedError

    @abc.abstractmethod
    def combine(self, aggregated_neighbor_tensor, self_tensor):
        raise NotImplementedError


class GraphConvolutionalLayer(GraphNeuralNetwork):

    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels)

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

    def aggregate(self, aggregated_neighbors_list):
        aggregated_neighbor_tensor = (
            torch.mean(torch.stack(aggregated_neighbors_list, dim=0), dim=0)
            if len(aggregated_neighbors_list) > 0
            else torch.zeros(self.in_channels)
        )
        return aggregated_neighbor_tensor

    def combine(self, aggregated_neighbor_tensor, self_tensor):
        result_tensor = self.f(self.w(aggregated_neighbor_tensor) + self.b(self_tensor))
        return result_tensor
