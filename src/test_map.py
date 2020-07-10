from layers import LPALayer
import tensorflow as tf

# create the tensorflow graph
class TEST_LPA(object):
    def __init__(self, args, initial_labels, adj):
        # self.args = args
        initial_labels = tf.constant(initial_labels, dtype=tf.int32)
        initial_labels = tf.Print(initial_labels, [initial_labels], "initial labels in the constructor", summarize=100)
        adj = tf.SparseTensor(*adj)
        list_ = []

        # temp = tf.argmax(self.outputs, axis=-1)
        # l0 = tf.one_hot(self.outputs_maxed, 6)
        l0 = initial_labels
        list_.append(l0)
        lp_layer = LPALayer(adj=adj)
        # l0 = tf.Print(l0, [l0], "labels current\n", summarize=100)
        # l0 = lp_layer(l0)
        # lp_layer = LPALayer(adj=adj)
        # l0 = tf.Print(l0, [l0], "labels current\n", summarize=100)
        # l0 = lp_layer(l0)
        # lp_layer = LPALayer(adj=adj)
        # l0 = tf.Print(l0, [l0], "labels current\n", summarize=100)
        # l0 = lp_layer(l0)
        for _ in range(3):
            
            # labels_curr = lp_layer(list_[-1])
            # l0 = tf.Print(l0, [l0], "labels current\n", summarize=100)
            l0 = lp_layer(list_[-1])
            list_.append(l0)

        self.outputs = list_
        # TODO
        # initialise the per_node lambdas with 


'''

A - B - C
| \ 
D  E

'''



model = TEST_LPA({}, [[0,0,1],[0,1,0],[1,0,0],[0,0,1],[0,0,1]],  [ [[0,1], [0,3], [0,4], [1,0], [1,2], [2,1], [3,0],[4,0]], [1,1,1,1,1,1,1,1],[5,5]])


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    outputs = sess.run([model.outputs])
    print(outputs)
