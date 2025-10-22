import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.readout import sum_nodes, broadcast_nodes, softmax_nodes


class Set2Set(nn.Module):
    def __init__(self, input_dim, n_iters, n_layers):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = th.nn.LSTM(self.output_dim, self.input_dim, n_layers)
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()

    def forward(self, graph, feat):
        with graph.local_scope():
            batch_size = graph.batch_size

            h = (feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
                 feat.new_zeros((self.n_layers, batch_size, self.input_dim)))

            q_star = feat.new_zeros(batch_size, self.output_dim)

            for _ in range(self.n_iters):
                q, h = self.lstm(q_star.unsqueeze(0), h)
                q = q.view(batch_size, self.input_dim)
                e = (feat * broadcast_nodes(graph, q)).sum(dim=-1, keepdim=True)
                graph.ndata['e'] = e
                alpha = softmax_nodes(graph, 'e')
                graph.ndata['r'] = feat * alpha
                readout = sum_nodes(graph, 'r')
                q_star = th.cat([q, readout], dim=-1)
            return q_star


class RESCAL(nn.Module):
    def __init__(self, rel_dim, graph_dim):
        super().__init__()
        self.n_features = rel_dim + graph_dim
        self.rel_proj = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.n_features, self.n_features * 2),
            nn.ELU(),
            nn.Linear(self.n_features * 2, graph_dim),
        )
        self.mlp = nn.Sequential(
            nn.Linear(graph_dim * 3, 2 * graph_dim),
            nn.LeakyReLU(),
            nn.Linear(2 * graph_dim, graph_dim),
            nn.LeakyReLU(),
            nn.Linear(graph_dim, 1),
        )

    def forward(self, heads, tails, combin, rels):
        rels = torch.cat((combin, rels), dim=1)
        rels = self.rel_proj(rels)

        rels = F.normalize(rels, dim=-1)
        heads = F.normalize(heads, dim=-1)
        tails = F.normalize(tails, dim=-1)

        scores= self.mlp(torch.cat([heads, tails,rels], dim=1))

        return scores


class PairReadout(nn.Module):
    def __init__(self, input_dim, n_iters, n_layers, rel_dim):
        super(PairReadout, self).__init__()
        self.graph_out1 = Set2Set(input_dim=input_dim, n_iters=n_iters, n_layers=n_layers)
        self.graph_out2 = Set2Set(input_dim=input_dim, n_iters=n_iters, n_layers=n_layers)
        self.graph_out3 = Set2Set(input_dim=input_dim, n_iters=n_iters, n_layers=n_layers)
        self.KGE = RESCAL(rel_dim=rel_dim, graph_dim=self.graph_out1.output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.graph_out1.reset_parameters()
        self.graph_out2.reset_parameters()
        self.graph_out3.reset_parameters()

    def forward(self, graph1, feat1, graph2, feat2, graph_cb, feat_cb, rel):
        emb1 = self.graph_out1(graph1, feat1)
        emb2 = self.graph_out2(graph2, feat2)
        emb3 = self.graph_out3(graph_cb, feat_cb)

        return self.KGE(emb1, emb2, emb3, rel)
