import argparse
import numpy as np
import tensorflow as tf
from time import time
from data_loader import load_data, load_npz, load_random
from train import train

seed = 234
np.random.seed(seed)
tf.set_random_seed(seed)

parser = argparse.ArgumentParser()

'''
# cora
parser.add_argument('--dataset', type=str, default='cora', help='which dataset to use')
parser.add_argument('--epochs', type=int, default=200, help='the number of epochs')
parser.add_argument('--dim', type=int, default=32, help='dimension of hidden layers')
parser.add_argument('--gcn_layer', type=int, default=5, help='number of GCN layers')
parser.add_argument('--lpa_iter', type=int, default=5, help='number of LPA iterations')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--lpa_weight', type=float, default=10, help='weight of LP regularization')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
'''

# # citeseer
parser.add_argument('--dataset', type=str, default='citeseer', help='which dataset to use')
parser.add_argument('--epochs', type=int, default=200, help='the number of epochs')
parser.add_argument('--dim', type=int, default=16, help='dimension of hidden layers')
parser.add_argument('--gcn_layer', type=int, default=2, help='number of GCN layers')
parser.add_argument('--lpa_iter', type=int, default=1, help='number of LPA iterations')
parser.add_argument('--l2_weight', type=float, default=5e-4, help='weight of l2 regularization')
parser.add_argument('--lpa_weight', type=float, default=1, help='weight of LP regularization')
parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.2, help='learning rate')

'''
# pubmed
parser.add_argument('--dataset', type=str, default='pubmed', help='which dataset to use')
parser.add_argument('--epochs', type=int, default=200, help='the number of epochs')
parser.add_argument('--dim', type=int, default=32, help='dimension of hidden layers')
parser.add_argument('--gcn_layer', type=int, default=2, help='number of GCN layers')
parser.add_argument('--lpa_iter', type=int, default=1, help='number of LPA iterations')
parser.add_argument('--l2_weight', type=float, default=2e-4, help='weight of l2 regularization')
parser.add_argument('--lpa_weight', type=float, default=1, help='weight of LP regularization')
parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
'''

'''
# coauthor-cs
parser.add_argument('--dataset', type=str, default='coauthor-cs', help='which dataset to use')
parser.add_argument('--epochs', type=int, default=200, help='the number of epochs')
parser.add_argument('--dim', type=int, default=32, help='dimension of hidden layers')
parser.add_argument('--gcn_layer', type=int, default=2, help='number of GCN layers')
parser.add_argument('--lpa_iter', type=int, default=2, help='number of LPA iterations')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--lpa_weight', type=float, default=2, help='weight of LP regularization')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
'''

'''
# coauthor-phy
parser.add_argument('--dataset', type=str, default='coauthor-phy', help='which dataset to use')
parser.add_argument('--epochs', type=int, default=200, help='the number of epochs')
parser.add_argument('--dim', type=int, default=32, help='dimension of hidden layers')
parser.add_argument('--gcn_layer', type=int, default=2, help='number of GCN layers')
parser.add_argument('--lpa_iter', type=int, default=3, help='number of LPA iterations')
parser.add_argument('--l2_weight', type=float, default=1e-4, help='weight of l2 regularization')
parser.add_argument('--lpa_weight', type=float, default=1, help='weight of LP regularization')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
'''

# '''
# random graph
# this is only for calculating the training time
# parser.add_argument('--dataset', type=str, default='random', help='which dataset to use')
# parser.add_argument('--epochs', type=int, default=100, help='the number of epochs')
# parser.add_argument('--dim', type=int, default=16, help='dimension of hidden layers')
# parser.add_argument('--gcn_layer', type=int, default=5, help='number of GCN layers')
# parser.add_argument('--lpa_iter', type=int, default=6, help='number of LPA iterations')
# parser.add_argument('--l2_weight', type=float, default=5e-8, help='weight of l2 regularization')
# parser.add_argument('--lpa_weight', type=float, default=15, help='weight of LP regularization')
# parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
# parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
# '''

t = time()
args = parser.parse_args()

if args.dataset in ['cora', 'citeseer', 'pubmed']:
    data = load_data(args.dataset)
elif args.dataset in ['coauthor-cs', 'coauthor-phy']:
    data = load_npz(args.dataset)
else:
    n_nodes = 100
    data = load_random(n_nodes=n_nodes, n_train=40, n_val=30, p=10/n_nodes)


# have some way to 
# change the % unknown
# change the size of data
# change the lp split


# labels is an N * labels
# 
features, labels, adj, train_mask, val_mask, test_mask = [data[i] for i in range(6)]

# adj
labels = labels[0:100]
train_mask = train_mask[0:100]
val_mask = val_mask[0:100]
test_mask = test_mask[0:100]

# adj[0] is the indices, then values and then shape
filter_indices = []
# new_adj_
for ele in adj[0]:
    if ele[0] < 100 and ele[1] < 100:
        filter_indices.append(True)
    else:
        filter_indices.append(False)

adj = (adj[0][filter_indices], adj[1][filter_indices], (100,100))
 
filter_indices = []
# new_adj_
for ele in features[0]:
    if ele[0] < 100:
        filter_indices.append(True)
    else:
        filter_indices.append(False)

features = (features[0][filter_indices], features[1][filter_indices], (100,3703))

# data = [features, labels, adj, train_mask, val_mask, test_mask]

# print(features.shape)
# print(labels)
# print(adj.shape)
# print(train_mask.shape)

train(args, data)
print('time used: %d s' % (time() - t))
