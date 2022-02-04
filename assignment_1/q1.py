import itertools

import numpy as np
import pandas as pd
import networkx as nx

from matplotlib import pyplot as plt
import seaborn as sns
import plotly.graph_objects as pg, plotly.express as px, plotly.figure_factory as pf


def load_graph_1():
    node_data = pd.read_csv("assignment_1/data/graph1.nodes")
    node_data = [(z, {"id": x, "name": y}) for x, y, z in node_data.values]

    with open("assignment_1/data/graph1.edges") as f:
        edge_data = list(
            map(lambda x: tuple(map(int, x.strip().split(","))), f.readlines())
        )
    edge_data = np.array(edge_data)

    g = nx.Graph()
    g.add_nodes_from(node_data)
    g.add_edges_from(edge_data)
    return g


def load_graph_2():
    with open("assignment_1/data/graph2.mtx") as f:
        edge_data = list(
            map(lambda x: tuple(map(int, x.strip().split(" "))), f.readlines()[1:])
        )
    edge_data = np.array(edge_data)

    g = nx.Graph()
    g.add_edges_from(edge_data)
    return g


def q1_1(g):
    print(g)

    degrees = np.array(g.degree)[:, 1]
    print("Max Degree:", np.max(degrees))
    print("Min Degree:", np.min(degrees))
    print("Average Degree:", np.mean(degrees))

    print("Graph Density:", nx.classes.function.density(g))
    print("Graph Sparsity:", 1 - nx.classes.function.density(g))

    fig = pf.create_distplot([degrees], group_labels=["node-degrees"])
    fig.show()


def q1_2(g):
    def find_cliques_size_k(g, k):
        all_cliques = set()
        for clique in nx.find_cliques(g):
            if len(clique) == k:
                all_cliques.add(tuple(sorted(clique)))
            elif len(clique) > k:
                for mini_clique in itertools.combinations(clique, k):
                    all_cliques.add(tuple(sorted(mini_clique)))
        return len(all_cliques)

    print("Number of 3-cliques", find_cliques_size_k(g, 3))
    print("Number of 4-cliques", find_cliques_size_k(g, 4))
    fig = pf.create_distplot(
        [np.array([len(clique) for clique in nx.find_cliques(g)])],
        group_labels=["clique-size"],
    )
    fig.show()

    eigenvector_centrality = nx.eigenvector_centrality_numpy(g)
    eigenvector_values = list(eigenvector_centrality.values())
    print("Mean Eigenvector Centrality", np.mean(eigenvector_values))
    print("Median Eigenvector Centrality", np.median(eigenvector_values))
    print("Max Eigenvector Centrality", np.max(eigenvector_values))
    print("Min Eigenvector Centrality", np.min(eigenvector_values))

    closeness_centrality = nx.closeness_centrality(g)
    closeness_values = list(closeness_centrality.values())
    print("Mean Closeness Centrality", np.mean(closeness_values))
    print("Median Closeness Centrality", np.median(closeness_values))
    print("Max Closeness Centrality", np.max(closeness_values))
    print("Min Closeness Centrality", np.min(closeness_values))

    fig = pf.create_distplot(
        [eigenvector_values, closeness_values],
        group_labels=["eigenvector-centrality", "closeness-centrality"],
    )
    fig.show()


def q1_3(g):
    # print("Mean Clustering Coefficient:", nx.algorithms.cluster.average_clustering(g))
    cluster_coeff = nx.algorithms.cluster.clustering(g)
    cluster_coeff = np.array(list(cluster_coeff.values()))
    print("Max Clustering Coefficient:", np.max(cluster_coeff))
    print("Min Clustering Coefficient:", np.min(cluster_coeff))
    print("Median Clustering Coefficient:", np.median(cluster_coeff))


def q1_4(g):
    print(len(list(nx.algorithms.bridges(g))))


def analyze(g):
    q1_4(g)


analyze(load_graph_1())
analyze(load_graph_2())
