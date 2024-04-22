# coding=utf-8
# Author: Jung
# Time: 2023/11/10 16:39


import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import dgl
# from dgl.nn import GraphConv
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from sklearn.cluster import KMeans
from Jung.evaluate import cluster_metrics
from Jung.utils import normalize_adj, sparse_mx_to_torch_sparse_tensor, normalize_adj_v2
np.random.seed(826)
torch.manual_seed(826)
torch.cuda.manual_seed(826)

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True, use_act = True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.ReLU()
        self.use_act = use_act
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.414)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj):
        seq_fts = self.fc(seq)
        out = torch.spmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        if self.use_act is True:
            return self.act(out)
        else:
            return out

def glorot_init(input_dim, output_dim):
    init_range = np.sqrt(6.0/(input_dim + output_dim))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return nn.Parameter(initial)

class GraphConvSparse(nn.Module):
    def __init__(self, input_dim, output_dim, activation = F.relu, **kwargs):
        super(GraphConvSparse, self).__init__(**kwargs)
        self.weight = glorot_init(input_dim, output_dim)
        self.activation = activation

    def forward(self, inputs, adj):
        x = inputs
        x = torch.mm(x,self.weight)
        x = torch.mm(adj, x)
        outputs = self.activation(x)
        return outputs


class VAGE(nn.Module):
    def __init__(self, graph, hid1_dim, hid2_dim):
        super(VAGE, self).__init__()
        self.graph = graph
        self.label = graph.ndata['label']
        self.feat = graph.ndata['feat'].to(torch.float32)
        self.feat_dim = self.feat.shape[1]
        self.base_gcn = GraphConvSparse(self.feat_dim, hid1_dim)
        self.gcn_mean = GraphConvSparse(hid1_dim, hid2_dim, activation = lambda x:x)
        self.gcn_logstddev = GraphConvSparse(hid1_dim, hid2_dim, activation = lambda x:x)
        self.hid2_dim = hid2_dim
        self.hid1_dim = hid1_dim
        self.adj = graph.adjacency_matrix().to_dense()
        self.adj = self.adj + torch.eye(self.graph.num_nodes())
        self.norm = self.adj.shape[0] * self.adj.shape[0] / float((self.adj.shape[0] * self.adj.shape[0] - self.adj.sum()) * 2)
        self.pos_weight = float(self.adj.shape[0] * self.adj.shape[0] - self.adj.sum()) / self.adj.sum()

        self.adj_1 = torch.from_numpy(normalize_adj(self.adj.numpy()).A)

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred

    def forward(self):
        hidden = self.base_gcn(self.feat, self.adj_1)
        self.mean = self.gcn_mean(hidden, self.adj_1)
        self.logstd = self.gcn_logstddev(hidden, self.adj_1)
        gaussian_noise = torch.randn(self.feat.size(0), self.hid2_dim)
        sampled_z = gaussian_noise * torch.exp(self.logstd) + self.mean
        A_pred = self.dot_product_decode(sampled_z)

        return sampled_z, A_pred


def load_data(name):
    r = "D:\\PyCharm_WORK\\MyCode\\Jung\\datasets\\attributed_nets\\"
    with open(r + name + ".pkl", 'rb') as f:
        data = pkl.load(f)
    graph = dgl.from_scipy(data['adj'])
    graph.ndata['feat'] = torch.from_numpy(data['feat'].todense())
    graph.ndata['label'] = torch.from_numpy(data['label'])
    return graph

"""
    Cora、Citeseer、Pubmed、Blogcatalog、ACM公用文件
"""
if __name__ == "__main__":

    graph = load_data("acm")

    model = VAGE(graph, 32, 16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # , weight_decay=0.10

    weight_mask = model.adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = model.pos_weight

    best_z = 0
    best_nmi = 0
    cm = cluster_metrics(model.label.numpy())
    for epoch in range(100):
        model.train()
        Z, A_pred = model()
        loss = log_lik = model.norm * F.binary_cross_entropy(A_pred.view(-1), (model.adj).view(-1), weight=weight_tensor)
        kl_divergence = 0.5 / A_pred.size(0) * (1 + 2 * model.logstd - model.mean ** 2 - torch.exp(model.logstd) ** 2).sum(1).mean()

        loss -= kl_divergence
        best_z = Z

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    acc_list = []
    f1_macro_list = []
    nmi_list = []
    ari_list = []

    for i in range(10):
        pred = KMeans(n_clusters=len(np.unique(model.label)), n_init=10).fit_predict(best_z.detach().numpy())
        acc_align, nmi, _, f1_macro, _, ari = cm.evaluation(pred)
        acc_list.append(acc_align)
        f1_macro_list.append(f1_macro)
        nmi_list.append(nmi)
        ari_list.append(ari)

        print(
            'nmi: {:.4f}, f1_score={:.4f},  acc = {:.4f}, ari= {:.4f}'.format(
                nmi,
                f1_macro,
                acc_align,
                ari
            ))
    print("nmi = {:.4f}+-{:.4f}, f1 = {:.4f}+-{:.4f}, acc = {:.4f}+-{:.4f}, ari = {:.4f}+-{:.4f}".format(
        np.mean(nmi_list), np.std(nmi_list),
        np.mean(f1_macro_list), np.std(f1_macro_list),
        np.mean(acc_list), np.std(acc_list),
        np.mean(ari_list), np.std(ari_list)
    ))