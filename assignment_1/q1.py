import numpy as np
import pandas as pd
import networkx as nx

from matplotlib import pyplot as plt
import seaborn as sns
import plotly.graph_objects as pg, plotly.express as px, plotly.figure_factory as pf


def load_graph_1():
    node_data = pd.read_csv('data/graph1.nodes')
    node_data = [(z, {"id": x, "name": y}) for x, y, z in node_data.values]

    with open('data/graph1.edges') as f:
        edge_data = list(map(lambda x: tuple(map(int, x.strip().split(','))), f.readlines()))
    edge_data = np.array(edge_data)

    g = nx.Graph()
    g.add_nodes_from(node_data)
    g.add_edges_from(edge_data)
    return g

def load_graph_2():
    with open('data/graph2.mtx') as f:
        edge_data = list(map(lambda x: tuple(map(int, x.strip().split(' '))), f.readlines()[1:]))
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
    print("Median Degree:", np.median(degrees))
    sns.displot(degrees)
    plt.show()

def q1_2(g):
    eigenvector_centrality = nx.eigenvector_centrality_numpy(g)
    eigenvector_values = list(eigenvector_centrality.values())

    closeness_centrality = nx.closeness_centrality(g)
    closeness_values = list(closeness_cent-rality.values())

    fig = pf.create_distplot(
        [eigenvector_values, closeness_values], 
        group_labels=['eigenvector-centrality', 'closeness-centrality']
    )
    fig.show()


def analyze(g):
    q1_1(g)


analyze(load_graph_1())
# analyze(load_graph_2())
