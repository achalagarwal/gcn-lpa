from layers import *
from inits import *


class GCN_LPA(object):
    def __init__(self, args, features, labels, adj):
        self.args = args
        # TODO
        # initialise the per_node lambdas with something small
        # self.per_node_lambdas = glorot(shape=(labels.shape[0],1))
        self.per_node_lambdas = tf.Variable(tf.zeros((labels.shape[0],1), dtype=tf.float64))
        self.vars = []  # for computing l2 loss

        self._build_inputs(features, labels)
        self._build_edges(adj)
        self._build_gcn(features[2][1], labels.shape[1], features[0].shape[0])
        self._build_lpa()
        self._build_train()
        self._build_eval()

    def _build_inputs(self, features, labels):
        self.features = tf.SparseTensor(*features)
        self.labels = tf.constant(labels, dtype=tf.int32)
        self.label_mask = tf.placeholder(tf.float64, shape=labels.shape[0])
        self.ver_label_mask = tf.placeholder(tf.float64, shape=labels.shape[0])
        self.test_label_mask = tf.placeholder(tf.float64, shape=labels.shape[0])
        self.dropout = tf.placeholder(tf.float64)

    def _build_edges(self, adj):
        edge_weights = glorot(shape=[adj[0].shape[0]])
        self.adj = tf.SparseTensor(adj[0], edge_weights, adj[2])
        # self.adj = tf.sparse_tensor_to_dense(self.adj)
        # self.adj = tf.Print(self.adj, [self.adj.indices], "adj graph", summarize=200)

        # self.adj = tf.SparseTensor()
        # self.adj_ = tf.sparse_to_dense(self.adj)
        # self.adj = tf.print(self.adj_.values, [self.adj_])
        self.normalized_adj = tf.sparse_softmax(self.adj)
        self.adj_lpa = tf.SparseTensor(adj[0], adj[1], adj[2])
        # tf.debugging.assert_non_positive(self.normalized_adj)

        self.vars.append(edge_weights)

    def _build_gcn(self, feature_dim, label_dim, feature_nnz):
        hidden_list = []

        if self.args.gcn_layer == 1:
            gcn_layer = GCNLayer(input_dim=feature_dim, output_dim=label_dim, adj=self.normalized_adj,
                                 dropout=self.dropout, sparse=True, feature_nnz=feature_nnz, act=lambda x: x)
            self.outputs = gcn_layer(self.features)
            self.vars.extend(gcn_layer.vars)
        else:
            gcn_layer = GCNLayer(input_dim=feature_dim, output_dim=self.args.dim, adj=self.normalized_adj,
                                 dropout=self.dropout, sparse=True, feature_nnz=feature_nnz)
            hidden = gcn_layer(self.features)
            hidden_list.append(hidden)
            self.vars.extend(gcn_layer.vars)

            for _ in range(self.args.gcn_layer - 2):
                gcn_layer = GCNLayer(input_dim=self.args.dim, output_dim=self.args.dim, adj=self.normalized_adj,
                                     dropout=self.dropout)
                hidden = gcn_layer(hidden_list[-1])
                hidden_list.append(hidden)
                self.vars.extend(gcn_layer.vars)

            gcn_layer = GCNLayer(input_dim=self.args.dim, output_dim=label_dim, adj=self.normalized_adj,
                                 dropout=self.dropout, act=lambda x: x)
            self.outputs = gcn_layer(hidden_list[-1])
            # self.per_node_lambdas = tf.Print(self.per_node_lambdas, [self.per_node_lambdas], message="per node lambdas",summarize=100 )
            # TODO
            # the per node lampdas do get trained, so it might not be smart to use a clip on them
            # how about a differentiable function?
            # self.per_node_lambdas = tf.clip_by_value(self.per_node_lambdas,-1,1)
            # self.vars.append(self.per_node_lambdas)
            # self.outputs = tf.Print(self.outputs, [self.outputs], message=" print outputs labels ",summarize=100 )
            # TODO
            # argmax is not differentiable, does this affect the flow of training?

            self.outputs_maxed = tf.argmax(self.outputs, axis=-1)
            self.outputs_maxed = tf.one_hot(self.outputs_maxed, 6)
            # self.outputs_maxed = tf.Print(self.outputs_maxed, [self.outputs_maxed], message=" print outputs maxed labels ",summarize=100 )
            # self.per_node_lambdas = tf.Print(self.per_node_lambdas, [self.per_node_lambdas], message="per node lambdas after clipping",summarize=100 )
            # self.outputs = self.outputs * self.per_node_lambdas
            
            # print(self.outputs.shape)
            self.vars.extend(gcn_layer.vars)

        self.prediction = tf.nn.softmax(self.outputs, axis=-1)

    def _build_lpa(self):
        label_mask = tf.expand_dims(self.label_mask, -1)
        input_labels = tf.cast(label_mask, dtype=tf.int32) * self.labels
        # input_labels = tf.Print(input_labels, [input_labels], message=" print input labels ",summarize=100 )
        label_list = [self.outputs_maxed]
        # label_list = [input_labels]

        for _ in range(self.args.lpa_iter):
            # lp_layer = LPALayer(adj=self.normalized_adj)
            lp_layer = LPALayer(adj=self.adj_lpa)
            hidden = lp_layer(label_list[-1])
            label_list.append(hidden)
        self.predicted_label = label_list[-1]
        # self.predicted_label = tf.cast(self.predicted_label, dtype=tf.float64) * (1- self.per_node_lambdas)
    def _build_train(self):
        # GCN loss

        self.label_mask_1 = tf.expand_dims(self.label_mask, -1)
        # self.per_node_lambdas = tf.clip_by_value(self.per_node_lambdas,-1,1)
        self.masked_lambdas = tf.where(self.label_mask > 0, self.per_node_lambdas , tf.zeros_like(self.per_node_lambdas))
        self.average_lambda = tf.reduce_sum(self.masked_lambdas) / tf.reduce_sum(self.label_mask)
        # self.masked_lambdas = tf.Print(self.masked_lambdas, [self.masked_lambdas], message="masked lambdas before the condition", summarize=200)
        # self.per_node_lambdas = tf.Print(self.per_node_lambdas, [self.per_node_lambdas], message="per node lambdas before the update", summarize=200)
        # print(self.per_node_lambdas.shape)
        self.average_lambda = tf.expand_dims(self.average_lambda, -1)
        self.average_lambda = tf.expand_dims(self.average_lambda, -1)
        # print(self.average_lambda.shape)
        shaper = tf.constant([self.per_node_lambdas.shape[0],1 ], tf.int32)
        self.average_tensor = tf.tile(self.average_lambda, shaper)
        # print(self.average_tensor.shape)

        # self.average_tensor = tf.constant(self.average_lambda, dtype=self.per_node_lambdas.dtype, shape=self.per_node_lambdas.shape, name='Const av_lambda')
        self.per_node_lambdas_1 = tf.where(self.masked_lambdas > 0, self.per_node_lambdas, self.average_tensor)
        # self.per_node_lambdas_1 = tf.Print(self.per_node_lambdas_1, [self.per_node_lambdas_1], message="per node lambdas after the update", summarize=200)

        # this is where the per_node lambdas get appended

        # this should be masked_lambdas as the average would affect the loss
        # also, maybe it might be better to just clip instead of regualarizing
        self.vars.append(self.per_node_lambdas_1*0.1)

        self.outputs = self.outputs * self.per_node_lambdas_1
        self.predicted_label = tf.cast(self.predicted_label, dtype=tf.float64) * (1- self.per_node_lambdas_1)

        self.final_outputs = self.outputs + self.predicted_label

        self.label_mask = tf.cast(self.label_mask, dtype=tf.float64)
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.final_outputs, labels=self.labels)
        self.train_loss = tf.reduce_sum(self.loss * self.label_mask) / tf.reduce_sum(self.label_mask)
        self.ver_loss = tf.reduce_sum(self.loss * self.ver_label_mask) / tf.reduce_sum(self.ver_label_mask)
        self.test_loss = tf.reduce_sum(self.loss * self.test_label_mask) / tf.reduce_sum(self.test_label_mask)
        # self.loss = self.loss * self.per_node_lambdas
        

        # LPA loss
        # lpa_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.predicted_label, labels=self.labels)
        # lpa_loss_train = tf.reduce_sum(tf.cast(lpa_loss, tf.float64) * self.label_mask) / tf.reduce_sum(self.label_mask)
        # lpa_loss_ver = tf.reduce_sum(tf.cast(lpa_loss, tf.float64) * self.ver_label_mask) / tf.reduce_sum(self.ver_label_mask)
        # lpa_loss_test = tf.reduce_sum(tf.cast(lpa_loss, tf.float64) * self.test_label_mask) / tf.reduce_sum(self.test_label_mask)

        # self.train_loss +=  lpa_loss_train
        # self.ver_loss += lpa_loss_ver
        # self.test_loss += lpa_loss_test


        self.moment = tf.nn.moments(self.per_node_lambdas_1, [0])
        
        # one parameter for every node
        # add the lambda and 1-lambda multiplication factor to each loss
        #  

        # L2 loss
        self.reg_loss = tf.zeros((),dtype=tf.dtypes.float64,name=None)

        for var in self.vars:
            self.reg_loss += self.args.l2_weight * tf.nn.l2_loss(var)

        self.train_loss += self.reg_loss
        self.test_loss += self.reg_loss
        self.ver_loss += self.reg_loss

        tf.summary.histogram("per_node_lambdas", self.per_node_lambdas_1)

        # these operations should only take place in the training stage 
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr).minimize(self.train_loss)


    def _build_eval(self):
        correct_prediction = tf.equal(tf.argmax(self.final_outputs, 1), tf.argmax(self.labels, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float64)
        self.train_accuracy = tf.reduce_sum(correct_prediction * self.label_mask) / tf.reduce_sum(self.label_mask)
        self.ver_accuracy = tf.reduce_sum(correct_prediction * self.ver_label_mask) / tf.reduce_sum(self.ver_label_mask)
        self.test_accuracy = tf.reduce_sum(correct_prediction * self.test_label_mask) / tf.reduce_sum(self.test_label_mask)


