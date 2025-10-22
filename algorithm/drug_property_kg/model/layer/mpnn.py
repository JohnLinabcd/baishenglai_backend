# -*- coding: utf-8 -*-
# @Time    : 2024/9/2 上午9:26
# @Author  : Chen Mukun
# @File    : kmpnn.py
# @Software: PyCharm
# @desc    : 


import torch
from torch import nn
from torch.nn import init

from dgl import function as fn
import torch.nn.functional as F
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair


class KGMPNNLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 attn_fc,
                 edge_func1,
                 edge_func2,
                 aggregator_type='mean',
                 residual=False,
                 bias=True):
        super(KGMPNNLayer, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.attn_fc = attn_fc
        self.edge_func1 = edge_func1
        self.edge_func2 = edge_func2
        if aggregator_type == 'sum':
            self.reducer = fn.sum
        elif aggregator_type == 'mean':
            self.reducer = fn.mean
        elif aggregator_type == 'max':
            self.reducer = fn.max
        else:
            raise KeyError('Aggregator type {} not recognized: '.format(aggregator_type))
        self._aggre_type = aggregator_type
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(self._in_dst_feats, out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_feats))
        else:
            self.register_buffer('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=init.calculate_gain('relu'))

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'attn_e': F.leaky_relu(a)}

    def message_func1(self, edges):
        return {'m1': edges.src['h'] * edges.data['w1'], 'attn_e1': edges.data['attn_e'], 'z1': edges.src['z']}

    def message_func2(self, edges):
        return {'m2': edges.src['h'] * edges.data['w2'], 'attn_e2': edges.data['attn_e'], 'z2': edges.src['z']}

    def reduce_func1(self, nodes):
        alpha = F.softmax(nodes.mailbox['attn_e1'], dim=1).unsqueeze(-1)
        h = torch.sum(alpha * nodes.mailbox['m1'], dim=1)
        return {'neigh1': h}

    def reduce_func2(self, nodes):
        alpha = F.softmax(nodes.mailbox['attn_e2'], dim=1).unsqueeze(-1)
        h = torch.sum(alpha * nodes.mailbox['m2'], dim=1)
        return {'neigh2': h}

    def forward(self, graph, feat, efeat):

        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)

            # (n, d_in, 1)
            graph.srcdata['h'] = feat_src.unsqueeze(-1)

            # (n, d_in, d_out)
            graph.edata['w1'] = self.edge_func1(efeat).view(-1, self._in_src_feats, self._out_feats)
            graph.edata['w2'] = self.edge_func2(efeat).view(-1, self._in_src_feats, self._out_feats)

            graph.ndata['z'] = feat_src
            graph.apply_edges(self.edge_attention)

            edges1 = torch.nonzero(graph.edata['etype'] == 0).squeeze(1).int()
            edges2 = torch.nonzero(graph.edata['etype'] == 1).squeeze(1).int()

            graph.send_and_recv(edges1, self.message_func1, self.reduce_func1)
            graph.send_and_recv(edges2, self.message_func2, self.reduce_func2)
            rst1 = graph.dstdata['neigh1'].sum(dim=1)
            rst2 = graph.dstdata['neigh2'].sum(dim=1)
            rst = rst1 + rst2

            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)

            if self.bias is not None:
                rst = rst + self.bias
            return rst


class KGMPNN(nn.Module):
    def __init__(self, args, entity_emb, relation_emb):
        super(KGMPNN, self).__init__()

        self.project_node_feats = nn.Sequential(
            nn.Linear(args['node_indim'], args['node_hidden_feats']),
            nn.ReLU()
        )
        self.num_step_message_passing = args['num_step_message_passing']
        attn_fc = nn.Linear(2 * args['node_hidden_feats'], 1, bias=False)
        edge_network1 = nn.Sequential(
            nn.Linear(args['edge_indim'], args['edge_hidden_feats']),
            nn.ReLU(),
            nn.Linear(args['edge_hidden_feats'], args['node_hidden_feats'] * args['node_hidden_feats'])
        )
        edge_network2 = nn.Sequential(
            nn.Linear(args['edge_indim'], args['edge_hidden_feats']),
            nn.ReLU(),
            nn.Linear(args['edge_hidden_feats'], args['node_hidden_feats'] * args['node_hidden_feats'])
        )

        self.gnn_layer = KGMPNNLayer(
            in_feats=args['node_hidden_feats'],
            out_feats=args['node_hidden_feats'],
            attn_fc=attn_fc,
            edge_func1=edge_network1,
            edge_func2=edge_network2,
            aggregator_type='sum'
        )

        self.gru = nn.GRU(args['node_hidden_feats'], args['node_hidden_feats'])
        self.out_dim = args['node_hidden_feats']

        atom_emb = torch.randn((118, args['node_indim']))
        node_emb = torch.cat((atom_emb, entity_emb), 0)
        bond_emb = torch.randn((4, args['edge_indim']))
        edge_emb = torch.cat((bond_emb, relation_emb), 0)
        self.node_emb = nn.Embedding.from_pretrained(node_emb, freeze=False)
        self.edge_emb = nn.Embedding.from_pretrained(edge_emb, freeze=False)

    def forward(self, g):
        node_feats = self.node_emb(g.ndata['h'])
        edge_feats = self.edge_emb(g.edata['e'])

        node_feats = self.project_node_feats(node_feats)  # (V, node_out_feats)
        hidden_feats = node_feats.unsqueeze(0)  # (1, V, node_out_feats)

        for _ in range(self.num_step_message_passing):
            node_feats = F.relu(self.gnn_layer(g, node_feats, edge_feats))
            node_feats, hidden_feats = self.gru(node_feats.unsqueeze(0), hidden_feats)
            node_feats = node_feats.squeeze(0)
        return node_feats
