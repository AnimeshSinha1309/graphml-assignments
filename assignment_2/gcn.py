import torch

from base import GraphNeuralNetwork


class GraphConvolutionalNetwork(GraphNeuralNetwork):
    def __init__(self, in_channels, hidden_channels, out_channels, layers):
        super().__init__(in_channels, hidden_channels, out_channels, layers)

        self.input_nn = torch.nn.Linear(
            in_channels,
            hidden_channels,
        )
        self.output_nn = torch.nn.Linear(
            hidden_channels,
            out_channels,
        )

        self.w = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    hidden_channels,
                    hidden_channels,
                    bias=False,
                )
                for _ in range(layers)
            ]
        )
        self.b = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    hidden_channels,
                    hidden_channels,
                    bias=False,
                )
                for _ in range(layers)
            ]
        )
        self.f = torch.nn.ReLU()

    def initialize(self, node_feature_tensor):
        return self.input_nn(node_feature_tensor)

    def aggregate(self, aggregated_neighbors_list, _layer_idx):
        aggregated_neighbor_tensor = (
            torch.mean(torch.stack(aggregated_neighbors_list, dim=0), dim=0)
            if len(aggregated_neighbors_list) > 0
            else torch.zeros(self.hidden_channels)
        )
        return aggregated_neighbor_tensor

    def combine(self, aggregated_neighbor_tensor, self_tensor, layer_idx):
        result_tensor = self.f(
            self.w[layer_idx](aggregated_neighbor_tensor)
            + self.b[layer_idx](self_tensor)
        )
        return result_tensor

    def output(self, node_feature_tensor):
        return self.output_nn(node_feature_tensor)
