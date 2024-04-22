# coding=utf-8
# Author: Jung
# Time: 2023/11/12 20:08
"""

    测试版VBPG, 增加簇内距离 / 修改优化算法

"""
import numpy as np
import scipy.sparse as sp
from Jung.utils import normalize_adj, sparse_mx_to_torch_sparse_tensor, normalize_adj_v2
import torch
import dgl
import pickle as pkl
import networkx as nx
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import wasserstein_distance

class VBPG(object):
    """
        邻接矩阵要求无环图
    """
    def __init__(self, adj, x=1):
        super(VBPG, self)
        self.x = x
        self.BOUNDARY = 2
        self.norn_adj = adj

        num_nodes = adj.shape[0]
        self.laplace = np.eye(num_nodes) - normalize_adj(adj).A
        self.dist = adj.sum(-1) / adj.sum()

        # has self-loop
        np.fill_diagonal(adj, 1)
        self.adj = torch.from_numpy(normalize_adj(adj).A) # D^(-0.5) @ A @ D^(-0.5) if not dgl

        self.x_list = [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5]

        self.x_index = 0

    def intra_dist(self, k, pre_label, feat):
        intra_distance = 0
        for i in range(k):
            sub_nodes = np.where(pre_label == i)
            if len(sub_nodes[0]) == 0:
                continue
            dp_i = feat[sub_nodes]
            dis = euclidean_distances(dp_i, dp_i)
            n_i = dp_i.shape[0]
            if n_i == 0 or n_i ==1:
                intra_distance = intra_distance
            else:
                intra_distance = intra_distance + 1 / k * 1 / (n_i * (n_i - 1)) * sum(sum(dis))
        if intra_distance ==0:
            self.dist = self.dist
        else:
            self.dist = self.dist * intra_distance

    def sinkhorn(self, K, dist, sin_iter):
        u = np.ones([len(dist), 1]) / len(dist)
        K_ = sp.diags(1. / dist) * K
        K_[np.isinf(K_)] = 0.
        dist = dist.reshape(-1, 1)
        for it in range(sin_iter):
            u = 1. / K_.dot(dist / (K.T.dot(u)))
            u[np.isinf(u)] = 0
        v = dist / (K.T.dot(u))
        v[np.isinf(v)] = 0
        delta = sp.diags(u.reshape(-1)) @ sp.csr_matrix(K) @ sp.diags(v.reshape(-1))
        return delta.A

    def mean_plug(self, delta_add, delta_dele, epsilon, sin_iter, flat):
        C = (self.x - 1) * self.laplace
        if flat is True:
            C = self.laplace
        K_add = np.exp((C * delta_add).sum() * C / epsilon)
        K_dele = np.exp((C * delta_dele).sum() * C / epsilon)
        delta_add = self.sinkhorn(K_add, self.dist, sin_iter)
        delta_dele = self.sinkhorn(K_dele, self.dist, sin_iter)
        return delta_add, delta_dele

    def var_plug(self, delta_add, delta_dele, epsilon, sin_iter, flat):
        C = (self.x-1) * self.laplace
        if flat is True:
            C = self.laplace
        K_add = np.exp(-(C * delta_add).sum() * C / epsilon)
        K_dele = np.exp(-(C * delta_dele).sum() * C / epsilon)
        delta_add = self.sinkhorn(K_add, self.dist, sin_iter)
        delta_dele = self.sinkhorn(K_dele, self.dist, sin_iter)
        return delta_add, delta_dele

    def update_limit(self, epoch, total):
        self.x = self.x - self.x * (epoch / total)
        print(self.x)

    def update_limit_by_index(self):
        self.x_index += 1
        self.x = self.x_list[self.x_index]

    def cal_mean(self, delta_add, delta_dele, epsilon, sin_iter, eta, k, pre_label, feat, flat = False):
        if flat == True:
            self.intra_dist(k, pre_label, feat)
        delta_add, delta_dele = self.mean_plug(delta_add, delta_dele, epsilon, sin_iter, flat)
        delta_add = delta_add * (self.BOUNDARY - self.x)
        delta_dele = delta_dele * self.x
        delta = (delta_dele - delta_add)
        delta =  eta * normalize_adj_v2(delta)
        delta = torch.from_numpy(delta.A)
        mean_adj = self.adj + delta
        return mean_adj, delta_add, delta_dele

    def cal_var(self, delta_add, delta_dele, epsilon, sin_iter, eta, k, pre_label, feat, flat = False):
        if flat == True:
            self.intra_dist(k, pre_label, feat)
        delta_add, delta_dele = self.var_plug(delta_add, delta_dele, epsilon, sin_iter,flat)
        delta_add = delta_add *  (self.BOUNDARY - self.x)
        delta_dele = delta_dele * self.x
        delta = (delta_add - delta_dele)
        delta =  eta * normalize_adj_v2(delta)
        delta = torch.from_numpy(delta.A)
        var_adj = self.adj + delta
        return var_adj, delta_add, delta_dele

if __name__ == "__main__":
    def load_data(name):
        r = "D:\\PyCharm_WORK\\MyCode\\Jung\\datasets\\"
        with open(r + name + ".pkl", 'rb') as f:
            data = pkl.load(f)
        graph = dgl.from_scipy(data['adj'])
        graph.ndata['feat'] = torch.from_numpy(data['feat'].todense())
        graph.ndata['label'] = torch.from_numpy(data['label'])
        return graph
    def sparse_to_tuple(sparse_mx):
        if not sp.isspmatrix_coo(sparse_mx):
            sparse_mx = sparse_mx.tocoo()
        coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
        values = sparse_mx.data
        shape = sparse_mx.shape
        return coords, values, shape
    graph = load_data("cora")
    plug = VBPG(graph.adjacency_matrix().to_dense().numpy(), x=1)
    delta = np.ones(graph.num_nodes()) * 0.5
    delta_add = delta
    delta_dele = delta
    delta_var_add = delta
    delta_var_dele = delta
    mean_adj_list = []
    var_adj_list = []


    for epoch in range(50):
        if epoch % 20 == 0:
            mean_adj, delta_add, delta_dele = plug.cal_mean(delta_add, delta_dele, epsilon=3, sin_iter=3, eta=1, k = 7, pre_label=graph.ndata['label'].numpy(), feat=graph.ndata['feat'].numpy(), flat = False if epoch <=0 else True )
            var_adj, delta_var_add, delta_var_dele = plug.cal_var(delta_var_add, delta_var_dele, epsilon=3, sin_iter=3, eta=0.2, k = 7, pre_label=graph.ndata['label'].numpy(), feat=graph.ndata['feat'].numpy(), flat = False if epoch <=0 else True)
            plug.update_limit(epoch, 100)
            print("update")
            mean_adj = sparse_to_tuple(sp.csc_matrix(mean_adj.numpy()))
            var_adj = sparse_to_tuple(sp.csc_matrix(var_adj.numpy()))
            mean_adj_list.append(mean_adj)
            var_adj_list.append(var_adj)
