import numpy as np
import tensorflow as tf
from model import GCN_LPA
import pickle

def print_statistics(features, labels, adj):
    n_nodes = features[2][0]
    n_edges = (len(adj[0]) - labels.shape[0]) // 2
    n_features = features[2][1]
    n_labels = labels.shape[1]
    labeled_node_rate = 20 * n_labels / n_nodes

    n_intra_class_edge = 0
    for i, j in adj[0]:
        if i < j and np.argmax(labels[i]) == np.argmax(labels[j]):
            n_intra_class_edge += 1
    intra_class_edge_rate = n_intra_class_edge / n_edges

    print('n_nodes: %d' % n_nodes)
    print('n_edges: %d' % n_edges)
    print('n_features: %d' % n_features)
    print('n_labels: %d' % n_labels)
    print('labeled node rate: %.4f' % labeled_node_rate)
    print('intra-class edge rate: %.4f' % intra_class_edge_rate)


def train(args, data, batch_test=False):
    features, labels, adj, train_mask, val_mask, test_mask = [data[i] for i in range(6)]
    labels = labels.astype(np.int32)
    # uncomment the next line if you want to print statistics of the current dataset
    # print_statistics(features, labels, adj)

    model = GCN_LPA(args, features, labels, adj)
    def get_feed_dict(mask, val_mask, test_mask, dropout):
        feed_dict = {model.label_mask: mask, model.test_label_mask: test_mask, model.ver_label_mask: val_mask,model.dropout: dropout}
        return feed_dict

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        best_val_acc = 0
        final_test_acc = 0
        lpa_labels, gcn_labels, lambdaz = [], [], []
        train_writer = tf.summary.FileWriter( './../logs/train_5 ', sess.graph)

        for epoch in range(args.epochs):
            # train
            # _, train_loss, train_acc = sess.run(
            #     [model.optimizer, model.loss, model.accuracy,], feed_dict=get_feed_dict(train_mask, args.dropout))
            merge = tf.summary.merge_all()


            summary,_, _, lpa_label, gcn_label, lambdas, moments, train_loss, test_loss, val_loss, train_acc, val_acc, test_acc  = sess.run(
                [merge, model.optimizer, model.loss,model.predicted_label, model.outputs, model.per_node_lambdas_1, model.moment, model.test_loss, model.train_loss, model.ver_loss, model.train_accuracy, model.test_accuracy, model.ver_accuracy], feed_dict=get_feed_dict(train_mask, val_mask, test_mask, args.dropout))
            
            train_writer.add_summary(summary, epoch)

            print(moments)
            lpa_labels.append(lpa_label)
            gcn_labels.append(gcn_label)
            lambdaz.append(lambdas)

            # validation
            # val_loss, val_acc = sess.run([model.loss, model.accuracy], feed_dict=get_feed_dict(val_mask, 0.0))

            # test
            # test_loss, test_acc = sess.run([model.loss, model.accuracy], feed_dict=get_feed_dict(test_mask, 0.0))

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc

            if not batch_test:
                print('epoch %d   train loss: %.4f  acc: %.4f   val loss: %.4f  acc: %.4f   test loss: %.4f  acc: %.4f'
                      % (epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))

        if not batch_test:
            print('final test acc: %.4f' % final_test_acc)
        else:
            return final_test_acc
        
        # pickle the three lists 
        filename = './data_stored'
        fileObject = open(filename, 'wb')
        pickle.dump((lpa_labels,gcn_labels,lambdaz), fileObject)
        fileObject.close()
