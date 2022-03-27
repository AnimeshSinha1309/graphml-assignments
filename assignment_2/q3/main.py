import argparse
import random

import numpy as np
import torch
import torch_geometric as pyg

from gcn import GraphConvolutionalNetwork
from rnn import RecurrentNeuralNetwork
from gin import GraphIsomorphismNetwork
from engine import training, rnn_train

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--task", default="citeseer", help="Choose the task (citeseer, gnncompare, rnn)"
    )
    parser.add_argument(
        "--layers", default=3, type=int, help="Number of Layers to be used in the GNN"
    )
    parser.add_argument(
        "--dims", default=10, type=int, help="The Dimentions of each layer in the GNN"
    )
    parser.add_argument(
        "--normalizer",
        default="row",
        type=str,
        help="How to normalize GNN aggregation (row, column, symmetric, none)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task == "citeseer":
        df = pyg.datasets.Planetoid("datasets/planetoid", "citeseer")
        network = GraphConvolutionalNetwork(
            df.data.x.shape[1],
            args.dims,
            7,
            args.layers,
            adj_normalizer=args.normalizer,
        )
        network = network.to(device)
        df.data = df.data.to(device)
        training(network, df.data)
    elif args.task == "gnncompare":
        df = pyg.datasets.Planetoid("datasets/planetoid", "citeseer")
        network = GraphIsomorphismNetwork(df.data.x.shape[1], args.dims, 7, args.layers)
        network = network.to(device)
        df.data = df.data.to(device)
        training(network, df.data)
    elif args.task == "rnn":
        rnn = RecurrentNeuralNetwork(1000, 100, 2)
        rnn_train(rnn)
