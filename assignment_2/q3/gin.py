import torch

from base import GraphNeuralNetwork


class GraphIsomorphismNetwork(GraphNeuralNetwork):
    def __init__(self, in_channels, hidden_channels, out_channels, layers):
        super().__init__(in_channels, hidden_channels, out_channels, layers, "none")

        self.input_nn = torch.nn.Linear(
            in_channels,
            hidden_channels,
        )
        self.output_nn = torch.nn.Linear(
            hidden_channels,
            out_channels,
        )

        self.eps = torch.nn.ModuleList([
            torch.nn.parameter.Parameter(torch.tensor(0.2))
            for _ in range(layers)
        ])
        self.f = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_channels, hidden_channels),
                )
                for _ in range(layers)
            ]
        )

    def initialize(self, node_feature_tensor):
        return self.input_nn(node_feature_tensor)

    def aggregate(self, aggregated_neighbors_list, _layer_idx):
        aggregated_neighbor_tensor = (
            torch.sum(torch.stack(aggregated_neighbors_list, dim=0), dim=0)
            if len(aggregated_neighbors_list) > 0
            else torch.zeros(self.hidden_channels)
        )
        return aggregated_neighbor_tensor

    def combine(self, aggregated_neighbor_tensor, self_tensor, layer_idx):
        result_tensor = self.f[layer_idx](
            aggregated_neighbor_tensor + (1 + self.eps[layer_idx]) * self_tensor
        )
        return result_tensor

    def output(self, node_feature_tensor):
        return torch.nn.Softmax(dim=1)(self.output_nn(node_feature_tensor))
