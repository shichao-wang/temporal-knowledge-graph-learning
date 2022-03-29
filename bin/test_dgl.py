import dgl
import torch
from dgl.udf import EdgeBatch, NodeBatch

graph = dgl.graph(([0, 0, 1], [1, 2, 2]))


def msg(edges: EdgeBatch):
    print("MSG", edges.edges())
    return {"h": torch.ones(10)}


def reduce(nodes: NodeBatch):
    print("RED", nodes.nodes(), nodes.mailbox["h"])
    return {}


graph.update_all(msg, reduce)
