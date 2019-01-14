# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import numpy as np
import tensorflow as tf
import argparse
from munkres import Munkres
import pickle
from sklearn.metrics import normalized_mutual_info_score


class DFC(object):
    def __init__(self, input_shape, filters, n_clusters, batch_size, reg, dimension, alpha, beta, lam, hyper_m, model_dir):
        self.input_shape = input_shape
        self.filters = filters
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.reg = reg
        self.dimension = dimension
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.hyper_m = hyper_m
        self.model_dir = model_dir
        self.weights = self._initialize_weights()
        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("clustering"))])

        self.recon_loss = None
        self.cluster_loss = None

        # placeholder
        self.x = tf.placeholder(tf.float32, [None, input_shape])

        # forward
        self.embedding = self.encoder(self.weights)
        self.decoder_net = self.decoder(self.embedding, self.weights)
        self.p = self.clustering_layer(self.weights)

        # loss function
        self.loss = self.build_loss(self.p, self.weights)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        self.init = tf.global_variables_initializer()
        self.sess = tf.InteractiveSession()

    def _initialize_weights(self):
        all_weights = dict()

        # encoder
        all_weights['enc_w1'] = tf.Variable(tf.truncated_normal(shape=[self.input_shape, self.filters[0]], stddev=0.02))
        all_weights['enc_w2'] = tf.Variable(tf.truncated_normal(shape=[self.filters[0], self.filters[1]], stddev=0.02))
        all_weights['enc_w3'] = tf.Variable(tf.truncated_normal(shape=[self.filters[1], self.filters[2]], stddev=0.02))
        all_weights['enc_w4'] = tf.Variable(tf.truncated_normal(shape=[self.filters[2], self.dimension], stddev=0.02))

        all_weights['enc_b1'] = tf.Variable(tf.zeros([self.filters[0]], dtype=tf.float32))
        all_weights['enc_b2'] = tf.Variable(tf.zeros([self.filters[1]], dtype=tf.float32))
        all_weights['enc_b3'] = tf.Variable(tf.zeros([self.filters[2]], dtype=tf.float32))
        all_weights['enc_b4'] = tf.Variable(tf.zeros([self.dimension], dtype=tf.float32))

        # decoder
        all_weights['dec_w1'] = tf.Variable(tf.truncated_normal(shape=[self.dimension, self.filters[2]], stddev=0.02))
        all_weights['dec_w2'] = tf.Variable(tf.truncated_normal(shape=[self.filters[2], self.filters[1]], stddev=0.02))
        all_weights['dec_w3'] = tf.Variable(tf.truncated_normal(shape=[self.filters[1], self.filters[0]], stddev=0.02))
        all_weights['dec_w4'] = tf.Variable(tf.truncated_normal(shape=[self.filters[0], self.input_shape], stddev=0.02))

        all_weights['dec_b1'] = tf.Variable(tf.zeros([self.filters[2]], dtype=tf.float32))
        all_weights['dec_b2'] = tf.Variable(tf.zeros([self.filters[1]], dtype=tf.float32))
        all_weights['dec_b3'] = tf.Variable(tf.zeros([self.filters[0]], dtype=tf.float32))
        all_weights['dec_b4'] = tf.Variable(tf.zeros([self.input_shape], dtype=tf.float32))

        all_weights['cluster'] = tf.Variable(tf.truncated_normal(shape=[self.n_clusters, 4], stddev=0.04, seed=2),
                                             name='clustering')

        return all_weights

    def encoder(self, weights):
        layer1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.x, weights['enc_w1']), weights['enc_b1']))
        layer2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(layer1, weights['enc_w2']), weights['enc_b2']))
        layer3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(layer2, weights['enc_w3']), weights['enc_b3']))
        layer4 = tf.nn.bias_add(tf.matmul(layer3, weights['enc_w4']), weights['enc_b4'])

        return layer4

    def decoder(self, x, weights):
        layer1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, weights['dec_w1']), weights['dec_b1']))
        layer2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(layer1, weights['dec_w2']), weights['dec_b2']))
        layer3 = tf.nn.relu(tf.nn.bias_add(tf.matmul(layer2, weights['dec_w3']), weights['dec_b3']))
        layer4 = tf.nn.bias_add(tf.matmul(layer3, weights['dec_w4']), weights['dec_b4'])

        return layer4

    def clustering_layer(self, weights):
        clusters = weights['cluster']
        times = 2.0 / (self.hyper_m - 1.0)
        p = 1 / tf.reduce_sum(tf.pow(tf.expand_dims(self.embedding, axis=1) - clusters, times), axis=2)
        p = tf.transpose(tf.transpose(p) / tf.reduce_sum(p, axis=1))
        return p

    def build_cluster_loss(self, p, weights):
        clusters = weights['cluster']
        p = tf.pow(p, self.hyper_m)
        cluster_loss = tf.multiply(p, tf.reduce_sum(tf.square(tf.expand_dims(self.embedding, axis=1) - clusters), axis=2))
        cluster_loss = tf.reduce_mean(tf.reduce_sum(cluster_loss))
        return cluster_loss

    def build_loss(self, p, weights):
        self.recon_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(self.decoder_net, self.x), 2.0)))
        self.cluster_loss = self.build_cluster_loss(p, weights)
        f_j = tf.reduce_sum(p, axis=0)
        self.entropy = tf.reduce_sum(tf.multiply(f_j, tf.log(f_j)))
        self.con_entropy = -1 * tf.reduce_sum(tf.multiply(p, tf.log(p)))
        loss = self.recon_loss + self.alpha * self.cluster_loss + self.beta * self.con_entropy + self.lam * self.entropy
        return loss

    def assign_var(self, value):
        self.sess.run(tf.assign(self.weights["cluster"], value))

    def get_cluster(self):
        return self.sess.run(self.weights["cluster"])

    def finetune_fit(self, X):
        total_loss, recon_loss, cluster_loss, _ = self.sess.run((
            self.loss,
            self.recon_loss,
            self.cluster_loss,
            self.optimizer),
            feed_dict={self.x: X}
        )
        return total_loss, recon_loss, cluster_loss

    def forward(self, X):
        features, targets = self.sess.run((
            self.embedding,
            self.p),
            feed_dict={self.x: X}
        )
        return features, targets

    def restore(self):
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters...')
            tf.reset_default_graph()
            self.sess.run(self.init)
            self.saver.recover_last_checkpoints(ckpt.all_model_checkpoint_paths)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint found, training from scratch...')
            tf.reset_default_graph()
            self.sess.run(self.init)


def best_map(L1, L2):
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate


def get_one_hot(a):
    n_class = a.max() + 1
    n_sample = a.shape[0]
    b = np.zeros((n_sample, n_class))
    b[:, a] = 1
    return b


def load_reuters(data_path='./data'):
    import os
    data = np.load(os.path.join(data_path, 'reutersidf10k.npy')).item()
    x = data['data']
    y = data['label']
    return x, y


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DFC')
    parser.add_argument('--n_clusters', default=4, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--steps', default=65, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--display_step', default=1, type=int)
    parser.add_argument('--dimension', default=4, type=int)
    args = parser.parse_args()

    # Data preprocessing
    Img, Label = load_reuters()
    print('Data preprocessing has done!')

    # Hyper parameters
    model_dir = 'pretrain'
    hyper_m = 2
    alpha = 1
    beta = 10000
    lam = 0

    # Initialize network
    model = DFC(
        input_shape=2000,
        filters=[500, 500, 2000],
        n_clusters=args.n_clusters,
        batch_size=args.batch_size,
        reg=None,
        dimension=args.dimension,
        alpha=alpha,
        beta=beta,
        lam=lam,
        hyper_m=hyper_m,
        model_dir=model_dir
    )
    model.restore()
    print('DFC-Net initlization has done!')

    # Initialize centers
    with open('centers/km_centers.pkl', 'rb') as f:
        centers = pickle.load(f)
    model.assign_var(centers)
    print('initialed asignments using embedding!')

    # train model
    for step in range(args.steps):
        start = step * args.batch_size
        if step + args.batch_size > Img.shape[0]:
            start = Img.shape[0] - args.batch_size

        # fit model
        Loss, recon_loss, cluster_loss = model.finetune_fit(
            Img[start:start+args.batch_size]
        )

        # test model
        features, targets = model.forward(Img)
        targets_label = np.argmax(targets, axis=1)
        missrate_xkl = err_rate(Label, targets_label)
        acc = 1 - missrate_xkl
        nmi = normalized_mutual_info_score(Label, targets_label)

        print("step: %d" % step, "acc: %.4f" % acc, "nmi: %.4f" % nmi)
        print("loss: %.4f" % Loss, "recon_loss: %.4f" % recon_loss, "cluster_loss: %.4f" % cluster_loss)

