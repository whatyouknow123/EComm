# encoding='utf-8'
import numpy as np
import tf_euler
import tensorflow as tf
from ..model import DeepWalk
from tensorflow.contrib.tensorboard.plugins import projector
from ..util import *
import os
import fire
from time import time
from tensorflow.python.platform import tf_logging as logging
tf.logging.set_verbosity(tf.logging.INFO)

# model args
walk_len = 40
walk_num = 1
win_size = 5
num_negs = 100

# training args
batch_size = 1024  # batch_size = node_num * pair_num
dim = 64
epoch = 10
save_steps = 1000
save_model_steps = 10000
lr = 0.01

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def init_graph(root_dir, date):
    # 1. Instantiation Euler Graph
    tf_euler.initialize_embedded_graph(os.path.join(root_dir, date))  # load data to euler graph


def obtain_model(root_dir, date):
    max_id = get_max_id(root_dir, date)  # import for embedding row number
    # 2. Instantiation Model
    model = DeepWalk(tf_euler.ALL_NODE_TYPE, [0, 1], max_id=max_id, dim=dim,
                     walk_len=walk_len, walk_num=walk_num, win_size=win_size,
                     num_negs=num_negs)
    return model


def summary_info(summary_writer, summary_dict, global_step):
    for tag, value in summary_dict.items():
        summary = tf.Summary()
        summary.value.add(tag=tag, simple_value=value)
        summary_writer.add_summary(summary, global_step)
    summary_writer.flush()


def config_embedding_projector(summary_writer):
    embed_config = projector.ProjectorConfig()
    embedding = embed_config.embeddings.add()
    embedding.metadata_path = RAW_EMBEDDING_META_FILE
    projector.visualize_embeddings(summary_writer, embed_config)


def train(root_dir, date, model_name):
    init_graph(root_dir, date)

    global_step = tf.train.get_or_create_global_step()

    max_id = get_max_id(root_dir, date)  # import for embedding row number

    model = obtain_model(root_dir, date)
    # 3. Construct Compute Graph
    # 3.1 Sample Batch

    batch_node_size = int(np.ceil(float(batch_size) / model.batch_size_ratio))
    batch_node_size = max(batch_node_size, 1)
    batch_node_size_num = (max_id+1) // batch_node_size
    # logging.info('batch_node_size={}, batch_size_ratio={}'.format(batch_node_size, model.batch_size_ratio))

    # source = tf_euler.sample_node(batch_node_size, tf_euler.ALL_NODE_TYPE)   # sample op
    # source.set_shape([batch_node_size])   # source is the set of starting node of each walk
    # source = tf.placeholder(tf.int64, shape=[None], name='source')

    dataset = tf.data.Dataset.from_tensor_slices(np.array(range(max_id+1), dtype=np.int64))
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_node_size)
    dataset = dataset.repeat(epoch)
    source = dataset.make_one_shot_iterator().get_next()

    # 3.2 Forward
    _, _, loss, metric_name, metric = model(source)

    # 3.3 Backward
    decayed_lr = tf.train.exponential_decay(lr, global_step, 50000, 0.96, staircase=True)
    optimizer = tf.train.AdamOptimizer(decayed_lr)
    train_op = optimizer.minimize(loss, global_step)

    beta1_power, beta2_power = optimizer._get_beta_accumulators()
    current_lr = (optimizer._lr_t * tf.sqrt(1 - beta2_power) / (1 - beta1_power))

    # 4. Start Running
    # num_steps = int((max_id + 1) // batch_node_size * epoch)  # flatten
    num_steps = epoch * batch_node_size_num
    logging.info("batch_node_size={}, batch_size_ratio={}, batch_num={}, epoch={}, num_steps={}".format(
        batch_node_size, model.batch_size_ratio, batch_node_size_num, epoch, num_steps))

    # hook
    # logging_hook = tf.train.LoggingTensorHook({'step': global_step,
    #                                   'loss': loss, metric_name: metric}, every_n_iter=save_steps)

    # summary_hook = tf.train.SummarySaverHook(output_dir=os.path.join(root_dir, date, CHECK_POINT_DIR, model_name),
    #                                         save_steps=save_steps, summary_op=[tf.summary.scalar(metric_name, metric),
    #                                                                             tf.summary.scalar('loss', loss)])
    saver_hook = tf.train.CheckpointSaverHook(os.path.join(root_dir, date, CHECK_POINT_DIR, model_name),
                                              save_steps=save_model_steps, saver=tf.train.Saver())

    early_stop_hook = tf.train.StopAtStepHook(num_steps=num_steps)

    with tf.train.MonitoredTrainingSession(hooks=[saver_hook, early_stop_hook],
                                           checkpoint_dir=os.path.join(root_dir, date, CHECK_POINT_DIR, model_name),
                                           log_step_count_steps=None,
                                           config=config) as sess:
        i = sess.run(global_step)  # init when restore form checkpoint
        logging.info('initial step={}'.format(i))

        summary_writer = tf.summary.FileWriter(os.path.join(root_dir, date, CHECK_POINT_DIR, model_name), sess.graph)
        config_embedding_projector(summary_writer)

        total_loss, total_metric, total_time = 0.0, 0.0, 0.0
        while not sess.should_stop():
            t1 = time()

            _, loss_v, metric_v = sess.run([train_op, loss, metric])

            total_time += time() - t1
            total_loss += loss_v
            total_metric += metric_v

            if (i + 1) % save_steps == 0:
                lr_v = sess.run(current_lr)
                avg_loss = total_loss / save_steps
                avg_metric = total_metric / save_steps
                logging.info("loss = {:.4f}, {:s} = {:.4f}, lr = {:.4f}, step = {:d} ({:.3f} sec)".format(
                    avg_loss, metric_name, avg_metric, lr_v, i+1, total_time))

                summary_info(summary_writer, {'loss': avg_loss, metric_name: avg_metric, 'lr': lr_v}, i+1)
                total_loss, total_metric, total_time = 0.0, 0.0, 0.0
            i += 1

        # for e in range(epoch):
        #     np.random.shuffle(all_ids)
        #     logging.info('epoch={}'.format(e))
        #     for i in range(batch_node_size_num + 1):
        #         if i == batch_node_size_num:
        #             batch = all_ids[i * batch_node_size:]
        #         else:
        #             batch = all_ids[i*batch_node_size: (i+1) * batch_node_size]
        #         sess.run(train_op, feed_dict={source: batch})


def test(root_dir, date, model_name):
    init_graph(root_dir, date)

    global_step = tf.train.get_or_create_global_step()

    model = obtain_model(root_dir, date)
    batch_node_size = int(np.ceil(float(batch_size) / model.batch_size_ratio))

    dataset = tf.data.TextLineDataset(os.path.join(root_dir, date, TEST_NODE_ID_FILE))
    dataset = dataset.map(
      lambda id_str: tf.string_to_number(id_str, out_type=tf.int64))
    dataset = dataset.batch(batch_node_size)
    source = dataset.make_one_shot_iterator().get_next()
    _, _, _, metric_name, metric = model(source)

    metric_vals = []
    with tf.train.MonitoredTrainingSession(
      checkpoint_dir=os.path.join(root_dir, date, CHECK_POINT_DIR, model_name),
      save_checkpoint_secs=None,
      log_step_count_steps=None,
      config=config) as sess:
        while not sess.should_stop():
            metric_val = sess.run(metric)
            # print('{}: {}'.format(metric_name, metric_val))
            metric_vals.append(metric_val)

    print("avg {} = {:.3f}".format(metric_name, np.mean(metric_vals)))


def save_embedding(root_dir, date, model_name, is_save_context):
    init_graph(root_dir, date)  # some info may rely on the graph

    global_step = tf.train.get_or_create_global_step()

    max_id = get_max_id(root_dir, date)
    model = obtain_model(root_dir, date)

    dataset = tf.data.Dataset.range(max_id+1)
    dataset = dataset.batch(batch_size)
    source = dataset.make_one_shot_iterator().get_next()
    embedding, context_embedding, _, _, _ = model(source)

    embedding_vals = []
    with tf.train.MonitoredTrainingSession(
      checkpoint_dir=os.path.join(root_dir, date, CHECK_POINT_DIR, model_name),
      save_checkpoint_secs=None,
      log_step_count_steps=None,
      config=config) as sess:
        while not sess.should_stop():
            if is_save_context:
                embedding_val = sess.run(context_embedding)
            else:
                embedding_val = sess.run(embedding)
            embedding_vals.append(embedding_val)

    embedding_val = np.concatenate(embedding_vals)
    file_name = NP_CONTEXT_EMBEDDING_FILE if is_save_context else NP_EMBEDDING_FILE
    np.save(os.path.join(root_dir, date, CHECK_POINT_DIR, model_name, file_name), embedding_val)
    print('save embedding done...')


def run(root_dir, date, model_name, mode='train', is_save_context=False):
    date = str(date)
    is_save_context = bool(is_save_context)
    logging.info("date={}, model_name={}, mode={}, is_save_context={}".format(date, model_name, mode, is_save_context))

    if mode == 'train':
        train(root_dir, date, model_name)
    elif mode == 'test':
        test(root_dir, date, model_name)
    elif mode == 'save_embedding':
        save_embedding(root_dir, date, model_name, is_save_context)
        # output_embedding(root_dir, date, model_name)


if __name__ == '__main__':
    fire.Fire(run)
