import abc

import torch

from engine import training


class GraphNeuralNetwork(torch.nn.Module, abc.ABC):
    def __init__(self, in_channels, hidden_channels, out_channels, layers, adj_normalizer="row"):
        super().__init__()

        assert adj_normalizer in ["row", "column", "symmetric", "none"]
        self.adj_normalizer = adj_normalizer

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.layers = layers

    def forward(self, x, edge_index):
        x = self.initialize(x)
        degrees = torch.zeros(size=(len(x),))

        for layer_idx in range(self.layers):
            aggregated_neighbors = [[] for _ in range(x.shape[0])]
            for edge in edge_index.T:
                aggregated_neighbors[edge[0]].append(
                    x[edge[1]]
                )
                degrees[edge[0]] += 1

            final_outputs = []
            for i in range(len(x)):
                aggregated_neighbor_tensor = self.aggregate(
                    aggregated_neighbors[i],
                    layer_idx
                )
                final_outputs.append(
                    self.combine(aggregated_neighbor_tensor, x[i], layer_idx,)
                )
            x = torch.stack(final_outputs, dim=0)
        x = self.output(x)
        return x

    def get_degree_normalizer(self, degree_u, degree_v):
        if self.adj_normalizer == "row":
            return degree_u
        if self.adj_normalizer == "column":
            return degree_v
        if self.adj_normalizer == "symmetric":
            return (degree_v * degree_u) ** 0.5
        if self.adj_normalizer == "none":
            return 1

    @abc.abstractmethod
    def initialize(self, node_feature_tensor):
        raise NotImplementedError

    @abc.abstractmethod
    def aggregate(self, aggregated_neighbors, layer_idx):
        raise NotImplementedError

    @abc.abstractmethod
    def combine(self, aggregated_neighbor_tensor, self_tensor, layer_idx):
        raise NotImplementedError

    @abc.abstractmethod
    def output(self, node_feature_tensor):
        raise NotImplementedError
