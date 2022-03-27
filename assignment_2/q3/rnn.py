import torch

from base import GraphNeuralNetwork
from engine import rnn_train


class RecurrentNeuralNetwork(GraphNeuralNetwork):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__(in_channels, hidden_channels, out_channels, -1)

        self.encoder_nn = torch.nn.Linear(
            in_channels,
            hidden_channels,
        )
        self.x_nn = torch.nn.Linear(
            hidden_channels,
            hidden_channels,
            bias=False,
        )
        self.h_nn = torch.nn.Linear(
            hidden_channels,
            hidden_channels,
        )
        self.output_nn = torch.nn.Linear(
            hidden_channels,
            out_channels,
        )
        self.activation = torch.nn.ReLU()
        self.output_activation = torch.nn.Softmax(dim=0)

    def forward(self, x):
        edge_index = torch.tensor([[i, i + 1] for i in range(x.shape[0])])
        self.layers = len(x)
        return super().forward(x, edge_index)

    def initialize(self, node_feature_tensor):
        node_feature_tensor = self.activation(self.encoder_nn(node_feature_tensor))
        dummy_type_feature_appendage = torch.zeros((node_feature_tensor.shape[0], 1))
        node_feature_tensor = torch.cat(
            (dummy_type_feature_appendage, node_feature_tensor), dim=1
        )
        new_node_feature_tensor = torch.zeros((1, node_feature_tensor.shape[1]))
        new_node_feature_tensor[0, 0] = 1
        final_feature_tensor = torch.cat(
            (new_node_feature_tensor, node_feature_tensor), dim=0
        )
        return final_feature_tensor

    def aggregate(self, aggregated_neighbors_list, _layer_idx):
        if len(aggregated_neighbors_list) == 0:
            return torch.zeros((self.hidden_channels + 1))
        assert len(aggregated_neighbors_list) == 1
        return aggregated_neighbors_list[0]

    def combine(self, neighbor_tensor, self_tensor, layer_idx):
        neighbor_type = neighbor_tensor[0]
        self_type = self_tensor[0]
        neighbor_tensor = neighbor_tensor[1:]
        self_tensor = self_tensor[1:]
        full_tensor = (
            self.x_nn(neighbor_tensor) + self.h_nn(self_tensor)
            if self_type == 1
            else neighbor_tensor
        )
        full_tensor = torch.cat((self_tensor[:1], full_tensor))
        return full_tensor

    def output(self, node_feature_tensor):
        node_feature_tensor = node_feature_tensor[0, 1:]
        output_tensor = self.output_activation(self.output_nn(node_feature_tensor))
        return output_tensor
