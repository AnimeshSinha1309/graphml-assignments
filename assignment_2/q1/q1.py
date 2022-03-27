import numpy as np


class Graph:
    def __init__(self, number_of_nodes, edge_list):
        self.num_nodes = number_of_nodes
        self.adj = [[] for i in range(self.num_nodes)]
        for u, v in edge_list:
            self.adj[u].append(v)
            self.adj[v].append(u)

        self.dist = np.full(shape=(number_of_nodes, number_of_nodes), fill_value=99999)
        for u, v in edge_list:
            self.dist[u, v] = 1
            self.dist[v, u] = 1
        for u in range(number_of_nodes):
            self.dist[u, u] = 0
        for k in range(number_of_nodes):
            for u in range(number_of_nodes):
                for v in range(number_of_nodes):
                    self.dist[u, v] = min(
                        self.dist[u, k] + self.dist[k, v], self.dist[u, v]
                    )

    def ring(self, u, k):
        ring_elements = []
        for i in range(self.num_nodes):
            if self.dist[u, i] == k:
                ring_elements.append(i)
        return ring_elements

    def degree_sequence(self, u, k):
        return list(map(lambda x: len(self.adj[x]), self.ring(u, k)))

    def layer_similarity(self, u, v, k):
        x = self.degree_sequence(u, k)
        y = self.degree_sequence(v, k)
        similarity_score = 0
        for i in range(min(len(x), len(y))):
            similarity_score += max(x[i], y[i]) / min(x[i], y[i]) - 1
        return similarity_score

    def similarity(self, u, v, k):
        if u == v:
            return 0
        answer = 0
        for i in range(k + 1):
            answer += self.layer_similarity(u, v, i)
        return answer + 0.00001

    def struct2vec_walk(self, start, count, distance, p_layer=0.2, max_layer=2):
        walks = []
        for i in range(count):
            walk = []
            node, layer = start, 0
            for iteration in range(distance):
                layer_shift = np.random.choice(
                    [0, +1, -1],
                    p=[
                        1 - p_layer,
                        0
                        if layer == max_layer
                        else p_layer
                        if layer == 0
                        else p_layer / 2,
                        0
                        if layer == 0
                        else p_layer
                        if layer == max_layer
                        else p_layer / 2,
                    ],
                )
                if layer_shift != 0:
                    layer = layer + layer_shift
                else:
                    p = np.array(
                        [self.similarity(node, j, layer) for j in range(self.num_nodes)]
                    )
                    p = p / np.sum(p)
                    node = np.random.choice(list(range(self.num_nodes)), p=p)
                walk.append(node)
            walks.append(walk)
        return walks

    def node2vec_walk(self, start, count, distance, p=0.8, q=1.5):
        walks = []
        for i in range(count):
            walk = []
            node = start
            previous_node = None
            for d in range(distance):
                weights = np.array(
                    [
                        1 / p
                        if previous_node is not None and v == previous_node
                        else 1 / q
                        if previous_node is not None
                        and self.dist[previous_node, v] == 1
                        else 1
                        for v in self.adj[node]
                    ]
                )
                walk.append(
                    np.random.choice(self.adj[node], p=weights / np.sum(weights))
                )
                previous_node = node
            walks.append(walk)
        return walks

    def deepwalk_walk(self, start, count, distance):
        walks = []
        for i in range(count):
            walk = []
            node = start
            for d in range(distance):
                walk.append(np.random.choice(self.adj[node]))
            walks.append(walk)
        return walks


g = Graph(
    number_of_nodes=8,
    edge_list=[
        [6, 2],
        [6, 4],
        [6, 5],
        [1, 4],
        [1, 5],
        [2, 3],
        [2, 4],
        [4, 5],
        [3, 7],
        [7, 0],
    ],
)

print(g.deepwalk_walk(5, 2, 3))
print(g.node2vec_walk(5, 2, 3))
print(g.struct2vec_walk(5, 2, 3))
