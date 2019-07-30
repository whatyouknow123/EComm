import tensorflow as tf
import tf_euler
from tf_euler.python import euler_ops
from tf_euler.python import layers


class DeepWalk(layers.Layer):
    def __init__(self, node_type, edge_type, max_id, dim, walk_len, walk_num, win_size, num_negs):
        super(DeepWalk, self).__init__()
        self.node_type = node_type
        self.edge_type = edge_type
        self.max_id = max_id
        self.num_negs = num_negs
        self.walk_len = walk_len
        self.walk_num = walk_num
        self.left_win_size = win_size
        self.right_win_size = win_size
        self.dim = dim

        # pair_num
        self.batch_size_ratio = \
            self.walk_num * int(euler_ops.gen_pair(tf.zeros([0, self.walk_len + 1], dtype=tf.int64),
                                                   self.left_win_size, self.right_win_size).shape[1])

        print('batch_size_ratio={}'.format(self.batch_size_ratio))

        self.target_encoder = layers.Embedding(self.max_id + 1, self.dim,
                                               initializer=lambda: tf.truncated_normal_initializer(
                                                stddev=1.0 / (self.dim ** 0.5)))
        self.context_encoder = layers.Embedding(self.max_id + 1, self.dim,
                                                initializer=lambda: tf.truncated_normal_initializer(
                                                 stddev=1.0 / (self.dim ** 0.5)))

    def call(self, inputs):
        src, pos, negs = self.sampler(inputs)
        embedding = self.target_encoder(src)
        embedding_pos = self.context_encoder(pos)
        embedding_negs = self.context_encoder(negs)
        loss, mrr = self.decoder(embedding, embedding_pos, embedding_negs)
        embedding = self.target_encoder(inputs)
        context_embedding = self.context_encoder(inputs)
        return embedding, context_embedding, loss, 'mrr', mrr

    def sampler(self, inputs):
        batch_size = tf.size(inputs)

        num_negs = self.num_negs

        src, pos, negs = [], [], []
        for i in range(self.walk_num):
            path = tf_euler.random_walk(inputs, [self.edge_type] * self.walk_len,
                                        default_node=self.max_id + 1)
            pair = tf_euler.gen_pair(path, self.left_win_size, self.right_win_size)
            num_pairs = pair.shape[1]
            src_, pos_ = tf.split(pair, [1, 1], axis=-1)

            negs_ = tf_euler.sample_node(batch_size * num_pairs * num_negs, self.node_type)

            src_ = tf.reshape(src_, [batch_size * num_pairs, 1])
            pos_ = tf.reshape(pos_, [batch_size * num_pairs, 1])

            negs_ = tf.reshape(negs_, [batch_size * num_pairs, num_negs])

            src.append(src_)
            pos.append(pos_)
            negs.append(negs_)

        if self.walk_num == 1:
            return src[0], pos[0], negs[0]  # no need to concat

        src = tf.concat(src, axis=0, name='src')
        pos = tf.concat(pos, axis=0, name='pos')
        negs = tf.concat(negs, axis=0, name='negs')
        return src, pos, negs

    def decoder(self, embedding, embedding_pos, embedding_negs):
        # batch_size = tf.cast(tf.shape(embedding)[0], tf.float32)   # batch_node_size * num_pair
        logits = tf.matmul(embedding, embedding_pos, transpose_b=True)
        neg_logits = tf.matmul(embedding, embedding_negs, transpose_b=True)
        true_xent = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.ones_like(logits), logits=logits)
        negative_xent = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.zeros_like(neg_logits), logits=neg_logits)  # labels=zeros, label=0/1; <=> label=+-1 log(-logits)
        loss = tf.reduce_sum(true_xent) + tf.reduce_sum(negative_xent)
        mrr = tf_euler.metrics.mrr_score(logits, neg_logits)
        return loss, mrr

    def compute_output_shape(self, input_shape):
        output_shape = input_shape.concatenate(self.dim)
        output_shape = [d if d is not None else -1 for d in output_shape.as_list()]
        return output_shape
