import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
import numpy as np

import inspect

from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import scatter, softmax, add_self_loops
from torch_geometric.nn.inits import glorot, zeros, uniform

from build_tree import build_stage_one_edges, build_stage_two_edges, build_cominbed_edges
from build_tree import build_icd9_tree, build_atc_tree


class OntologyEmbedding(nn.Module):
    def __init__(self, voc, build_tree_func,
                 in_channels=100, out_channels=20, head=5):
        super(OntologyEmbedding, self).__init__()

        # initial tree edges
        res, graph_voc = build_tree_func(list(voc.idx2word.values()))
        stage_one_edges = build_stage_one_edges(res, graph_voc)
        stage_two_edges = build_stage_two_edges(res, graph_voc)

        self.edges1 = torch.tensor(stage_one_edges)
        self.edges2 = torch.tensor(stage_two_edges)
        self.graph_voc = graph_voc

        # construct model
        assert in_channels == head * out_channels
        # self.g = GATConv(in_channels=in_channels,
        #                  out_channels=out_channels,
        #                  heads=heads)

        self.g = GATv2(in_channels=in_channels,
                         out_channels=out_channels,
                         head=head,
                         num_layers=3)

        # tree embedding
        num_nodes = len(graph_voc.word2idx)
        self.embedding = nn.Parameter(torch.Tensor(num_nodes, in_channels))

        # idx mapping: FROM leaf node in graphvoc TO voc
        self.idx_mapping = [self.graph_voc.word2idx[word]
                            for word in voc.idx2word.values()]

        self.init_params()

    def get_all_graph_emb(self):
        emb = self.embedding
        emb = self.g(self.g(emb, self.edges1.to(emb.device)),
                     self.edges2.to(emb.device))
        return emb

    def forward(self):
        """
        :param idxs: [N, L]
        :return:
        """
        emb = self.embedding

        emb = self.g(self.g(emb, self.edges1.to(emb.device)),
                     self.edges2.to(emb.device))

        return emb[self.idx_mapping]

    def init_params(self):
        glorot(self.embedding)


# class GATv2(nn.Module):
#     def __init__(self,
#                 in_channels,
#                 out_channels,
#                 head=1,
#                 concat=True,
#                 negative_slope=0.2,
#                 dropout=0,
#                 bias=True):
#         super(GATv2, self).__init__()

#         self.gat = GATv2Conv(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             heads=head,
#             concat=concat,
#             negative_slope=negative_slope,
#             dropout=dropout,
#             bias=bias)

#     def forward(self, x, edge_index):
#         x = self.gat(x, edge_index)
#         return x


#     def __repr__(self):
#         return '{}({}, {}, heads={})'.format(self.__class__.__name__,
#                                              self.in_channels,
#                                              self.out_channels, self.head)

class GATv2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 head=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0.1,
                 bias=True,
                 num_layers=2):
        super(GATv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.head = head
        self.concat = concat
        self.num_layers = num_layers

        layers = []

        for i in range(num_layers):
            in_dim = in_channels if i == 0 else out_channels * head if concat else out_channels
            out_dim = out_channels

            layers.append(
                GATv2Conv(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    heads=head,
                    concat=concat,
                    negative_slope=negative_slope,
                    dropout=dropout,
                    bias=bias
                )
            )

        self.gat = nn.ModuleList(layers)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        for layer in self.gat:
            x = layer(x, edge_index)
            x = self.activation(x)
        return x

    def __repr__(self):
        return '{}({}, {}, heads={}, layers={})'.format(self.__class__.__name__,
                                                        self.in_channels,
                                                        self.out_channels,
                                                        self.head,
                                                        self.num_layers)

class ConcatEmbeddings(nn.Module):
    """Concat rx and dx ontology embedding for easy access
    """

    def __init__(self, config, dx_voc, rx_voc):
        super(ConcatEmbeddings, self).__init__()
        # special token: "[PAD]", "[CLS]", "[MASK]"
        self.special_embedding = nn.Parameter(
            torch.Tensor(config.vocab_size - len(dx_voc.idx2word) - len(rx_voc.idx2word), config.hidden_size))
        self.rx_embedding = OntologyEmbedding(rx_voc, build_atc_tree,
                                              config.hidden_size, config.graph_hidden_size,
                                              config.graph_heads)
        self.dx_embedding = OntologyEmbedding(dx_voc, build_icd9_tree,
                                              config.hidden_size, config.graph_hidden_size,
                                              config.graph_heads)
        self.init_params()

    def forward(self, input_ids):
        emb = torch.cat(
            [self.special_embedding, self.rx_embedding(), self.dx_embedding()], dim=0)
        return emb[input_ids]

    def init_params(self):
        glorot(self.special_embedding)


class FuseEmbeddings(nn.Module):
    """Construct the embeddings from ontology, patient info and type embeddings.
    """

    def __init__(self, config, dx_voc, rx_voc):
        super(FuseEmbeddings, self).__init__()
        self.ontology_embedding = ConcatEmbeddings(config, dx_voc, rx_voc)
        self.type_embedding = nn.Embedding(2, config.hidden_size)
        # self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

    def forward(self, input_ids, input_types=None, input_positions=None):
        """
        :param input_ids: [B, L]
        :param input_types: [B, L]
        :param input_positions:
        :return:
        """
        # return self.ontology_embedding(input_ids)
        ontology_embedding = self.ontology_embedding(
            input_ids) + self.type_embedding(input_types)
        return ontology_embedding
