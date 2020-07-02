from inits import *
import tensorflow as tf
from abc import abstractmethod

_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    random_tensor = keep_prob
    random_tensor += tf.random_uniform([noise_shape], dtype=tf.float64)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    res = tf.sparse_retain(x, dropout_mask)
    res /= keep_prob
    return res


def dot(x, y, sparse):
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    def __init__(self, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = []

    @abstractmethod
    def _call(self, inputs):
        pass

    def __call__(self, inputs):
        outputs = self._call(inputs)
        return outputs


class GCNLayer(Layer):
    def __init__(self, input_dim, output_dim, adj, dropout, sparse=False, feature_nnz=0, act=tf.nn.relu, name=None):
        super(GCNLayer, self).__init__(name)
        self.adj = adj
        self.dropout = dropout
        self.sparse = sparse
        self.feature_nnz = feature_nnz
        self.act = act
        with tf.variable_scope(self.name):
            self.weights = glorot([input_dim, output_dim], name='weight')
            self.vars = [self.weights]

    def _call(self, inputs):
        x = inputs
        x = sparse_dropout(x, 1 - self.dropout, self.feature_nnz) if self.sparse else tf.nn.dropout(x, 1 - self.dropout)
        x = dot(x, self.weights, sparse=self.sparse)
        x = dot(self.adj, x, sparse=True)
        return self.act(x)


class LPALayer(Layer):
    def __init__(self, adj, name=None):
        super(LPALayer, self).__init__(name)
        self.adj = adj

    def _call(self, inputs):

        # call a function here with both the adj and the inputs
        # the inputs == labels
        # the adj == adjacency (change the normalisation of this when its called)

        # in the function create another function which returns the map_function
        # the map_function (first one) will take the dense adjacency matrix and for each element multiply the value with the
        # value in the tensor
        # thus the labels are sent to the adj matrix

        # now we just need to get the frequency mode
        # tf.unique_with_counts
        # arg max on the counts
        # this is the map for each element of the intermediate adj matrix
        # so the adj matrix gets compressed to a ?x1 tensor which is then returned
        with tf.device('/cpu:0'):
            return mapper(self.adj, inputs)
        # output = dot(self.adj, inputs, sparse=True)
        # return output

with tf.device('/cpu:0'):
    def mapper(adj, labels):
        
        print(adj.shape)
        print(labels.shape)
        # this function has to return the new labels

        # create a new matrix of the same size as adj but with the indices instead of edge labels
        # what if its -1?

        adj_dense = tf.sparse.to_dense(adj)
        adj_dense = tf.cast(adj_dense, dtype=tf.int64)
        labels = tf.cast(labels, dtype=tf.float32)

        # adj_cols = tf.expand_dims(tf.range(tf.shape(adj_dense[0])[0], dtype=tf.int32), 1)
        adj_cols = tf.range(tf.shape(adj_dense[0])[0], dtype=tf.int32)
        


        # this function has to map each element of the dense adj matrix to the corresponding label
        # in the labels matrix
        def _mapper():
            
            def _element_wise_map(row):
                
                # print(row.shape)
                # adj_cols is the index map i.e. enumerate
                return tf.map_fn(lambda x: tf.cast(tf.argmax(labels[x[1]]) * x[0], dtype=tf.int32),(row, adj_cols), dtype=tf.int32)

            return _element_wise_map

        def util_map(label_row):
            # print(label_row.shape)
            lr = tf.unique_with_counts(label_row)
        
            return tf.cast(lr.y[tf.argmax(lr.count)], dtype=tf.int32)

        # so mapper is going to return the label adjacency matrix
        fn = _mapper()
        
        new_labels = tf.map_fn(lambda row: util_map(fn(row)), adj_dense, dtype=tf.int32)

        return tf.one_hot(new_labels, 6)