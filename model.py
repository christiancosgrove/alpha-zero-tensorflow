import tensorflow as tf
from tensorflow import layers
import numpy as np
import os
import pickle
import time

class Model:
    def __init__(self, width, height, sess=None, gpu_usage=None):
        self.width = width
        self.height = height

        self.mb_size = 64

        self.is_training = tf.placeholder(tf.bool)

        self.channels = 2
        self.X = tf.placeholder(tf.float32, shape=(None, self.width, self.height, self.channels))
        regularizer = tf.contrib.layers.l2_regularizer(1e-5)

        conv = tf.nn.relu(self.bn(tf.layers.conv2d(self.X, 64, 3, kernel_regularizer=regularizer)))
        conv = self.res_block(conv, 64)
        conv = self.res_block(conv, 64)
        conv = self.res_block(conv, 64)
        conv = self.res_block(conv, 64)
        self.policy = tf.nn.relu(self.bn(tf.layers.conv2d(conv, 2, 1, kernel_regularizer=regularizer)))
        s = self.policy.get_shape()
        self.policy =  tf.reshape(self.policy, (-1, int(s[1]*s[2]*s[3])))
        self.policy = tf.layers.dense(inputs=self.policy, units=self.width)
        self.policy_softmax = tf.nn.softmax(self.policy)

        self.value = tf.nn.relu(self.bn(tf.layers.conv2d(conv, 1, 1, kernel_regularizer=regularizer)))
        s = self.value.get_shape()
        self.value =  tf.reshape(self.value, (-1, int(s[1]*s[2]*s[3])))
        print('v', s)
        self.value = tf.nn.relu(tf.layers.dense(inputs=self.value, units=256, kernel_regularizer=regularizer))
        self.value = tf.layers.dense(inputs=self.value, units=1, activation=tf.nn.tanh, kernel_regularizer=regularizer)
        print('v', self.value.get_shape())

        self.policy_target = tf.placeholder(tf.float32, shape=(None, self.width))
        self.value_target = tf.placeholder(tf.float32, shape=(None, 1))

        self.loss_policy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.policy, labels=self.policy_target))
        self.loss_value = tf.losses.mean_squared_error(self.value,self.value_target, reduction=tf.losses.Reduction.MEAN)
        self.loss = self.loss_value + self.loss_policy

        tf.summary.scalar('loss_value', self.loss_value)
        tf.summary.scalar('loss_policy', self.loss_policy)

        self.buffer_size = 20*10000
        self.curr_buffer_position = 0
        self.X_dat = np.zeros((self.buffer_size, self.width, self.height, self.channels))
        self.policy_dat = np.zeros((self.buffer_size, self.width))
        self.value_dat = np.zeros((self.buffer_size, 1))
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # self.minimize_op = tf.train.MomentumOptimizer(learning_rate=1e-3, momentum=0.9).minimize(self.loss)
            self.minimize_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)
            # self.minimize_op = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5, beta2=0.9).minimize(self.loss)

        # if gpu_usage is not None:
        gpu_options = tf.GPUOptions(allow_growth=True)#per_process_gpu_memory_fraction=gpu_usage)

        if sess is None:
            self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        else:
            self.sess = sess
        # else:
        #     self.sess = tf.Session()
            
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        self.writer = tf.summary.FileWriter('logs')
        self.summaries = tf.summary.merge_all()

        self.step = 0

    def bn(self, X):
        return tf.layers.batch_normalization(X, training=self.is_training)

    def res_block(self, X, num_filters=16):
        regularizer = tf.contrib.layers.l2_regularizer(1e-5)
        conv = tf.nn.relu(self.bn(tf.layers.conv2d(X, num_filters, 3, kernel_regularizer=regularizer, padding='same')))
        conv = self.bn(tf.layers.conv2d(conv, num_filters, 3, kernel_regularizer=regularizer, padding='same'))
        conv += X
        return tf.nn.relu(conv)


    def clear_buffer(self):
        self.curr_buffer_position = 0

    def predict(self, X):
        return self.sess.run([self.policy_softmax, self.value], feed_dict={self.X: X, self.is_training:False})
    
    def add_data(self, X, policy_target, value_target):

        size = X.shape[0]
        if self.curr_buffer_position + X.shape[0] >= self.buffer_size:
            size = self.buffer_size - (self.curr_buffer_position)

        self.X_dat[self.curr_buffer_position : self.curr_buffer_position + size, :, :, :] = X[:size]
        self.policy_dat[self.curr_buffer_position : self.curr_buffer_position + size, :] = policy_target[:size]
        self.value_dat[self.curr_buffer_position : self.curr_buffer_position + size, :] = value_target[:size]
        self.curr_buffer_position += size
        if self.curr_buffer_position >= self.buffer_size:
            self.curr_buffer_position -= self.buffer_size

    def evaluate(self):
        lp = 0
        lv = 0
        x = 0
        while x < self.curr_buffer_position:
            loss_policy, loss_value = self.sess.run([self.loss_policy,self.loss_value], feed_dict={
                self.X:self.X_dat[x:x+self.mb_size],
                self.policy_target:self.policy_dat[x:x+self.mb_size],
                self.value_target:self.value_dat[x:x+self.mb_size],
                self.is_training:False})
            lp += loss_policy
            lv += loss_value
            x += self.mb_size
            
        return lp/x*self.mb_size, lv/x*self.mb_size

    def load_buffer(self, data_dir='data'):
        self.clear_buffer()
        for filename in os.listdir(data_dir):
            with open(os.path.join(data_dir, filename), 'rb') as file:
                try:
                    states, probs, values = pickle.load(file)
                    self.add_data(states, probs, values)
                except EOFError:
                    pass
    def train(self):
        print('training')
        
        while True:
            self.load_buffer()
            if self.curr_buffer_position < 1000:
                time.sleep(60)
            else:
                self.save()
                inds = np.random.permutation(self.curr_buffer_position)
                for idx in range(inds.size // self.mb_size):
                    indices = inds[idx * self.mb_size : (idx + 1) * self.mb_size]
                    _, l, summ = self.sess.run([self.minimize_op, self.loss, self.summaries], feed_dict={
                        self.X:self.X_dat[indices],
                        self.policy_target:self.policy_dat[indices],
                        self.value_target:self.value_dat[indices],
                        self.is_training:True})
                    if idx % 500 == 0:
                        self.writer.add_summary(summ, global_step=self.step)
                    if idx % 5000 == 0:
                        print(self.evaluate())

                    

                    self.step += 1
    def restore(self, checkpoints_dir='checkpoints'):
        self.restore_from_file(tf.train.latest_checkpoint(checkpoints_dir))

    def restore_from_file(self, filename):
        self.saver.restore(self.sess, save_path=filename)

    def save(self, checkpoints_dir='checkpoints'):
        self.saver.save(sess=self.sess,save_path=os.path.join(checkpoints_dir, 'checkpoint'), global_step=self.step)
    