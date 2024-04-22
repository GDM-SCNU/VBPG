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
from VariationalPG.vbpg import VBPG
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

    def forward(self, mean_adj, var_adj):
        hidden = self.base_gcn(self.feat, self.adj_1)
        self.mean = self.gcn_mean(hidden, mean_adj)
        self.logstd = self.gcn_logstddev(hidden, var_adj)
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


if __name__ == "__main__":

    graph = load_data("cora")
    model = VAGE(graph, 32, 16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # , weight_decay=0.10
    # nmi = 0.4925+-0.0000, f1 = 0.3735+-0.2395, acc = 0.8149+-0.0000, ari = 0.5139+-0.0000
    weight_mask = model.adj.view(-1) == 1
    weight_tensor = torch.ones(weight_mask.size(0))
    weight_tensor[weight_mask] = model.pos_weight

    best_z = 0
    best_nmi = 0

    delta = np.ones(graph.num_nodes()) * 0.5
    delta_add = delta
    delta_dele = delta
    delta_var_add = delta
    delta_var_dele = delta
    mean_adj = model.adj_1
    var_adj = model.adj_1
    plug = VBPG(graph.adjacency_matrix().to_dense().numpy(), x=1)
    best_nmi = 0
    best_pred = 0
    cm = cluster_metrics(model.label.numpy())
    update_dis = False
    for epoch in range(100):
        model.train()
        Z, A_pred = model(mean_adj, var_adj)
        loss = log_lik = model.norm * F.binary_cross_entropy(A_pred.view(-1), (model.adj).view(-1), weight=weight_tensor)
        kl_divergence = 0.5 / A_pred.size(0) * (1 + 2 * model.logstd - model.mean ** 2 - torch.exp(model.logstd) ** 2).sum(1).mean()

        loss -= kl_divergence
        best_z = Z

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = KMeans(n_clusters=len(np.unique(model.label)), n_init=10).fit_predict(best_z.detach().numpy())
        _, nmi, _, _, _, _ = cm.evaluation(pred)
        if nmi > best_nmi:
            best_nmi = nmi
            best_pred = pred
            update_dis = True
        if epoch % 20 == 0:
            if epoch == 0:
                update_dis = False
            mean_adj, delta_add, delta_dele = plug.cal_mean(delta_add, delta_dele, epsilon=3, sin_iter=2, eta=0.9,
                                                            k=len(np.unique(model.label)), pre_label=best_pred,
                                                            feat=model.feat.numpy(), flat=update_dis)
            var_adj, delta_var_add, delta_var_dele = plug.cal_var(delta_var_add, delta_var_dele, epsilon=3, sin_iter=2,
                                                                  eta=0.6, k=len(np.unique(model.label)), pre_label=best_pred,
                                                                  feat=model.feat.numpy(),
                                                                  flat=update_dis)
            mean_adj = mean_adj.float()
            var_adj = var_adj.float()
            # plug.update_limit(epoch, 500)
            plug.update_limit_by_index()
            print("update")
            update_dis = False

    acc_list = []
    f1_macro_list = []
    nmi_list = []
    ari_list = []
    cm = cluster_metrics(model.label.numpy())

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