import tensorflow as tf
slim = tf.contrib.slim
from tensorflow.contrib.slim.nets import resnet_v2
tf_model_nets = tf.contrib.slim.nets
import numpy as np
import evaluation as evl


class model:
    def __init__(self, config):
        self.device = config['device']
        self.label_dim = config['label_dim']  # Dim of label.
        self.codes_dim = config['codes_dim']  # Dim of codes.
        self.batch_size = config['batch_size']
        self.n_train = config['n_train']
        self.n_database = config['n_database']
        self.n_query = config['n_query']
        self.max_epoch = config['max_epoch']
        self.theta = config['theta']  # Parameter theta.
        self.lamda = config['lamda']  # Parameter lambda.
        self.alpha = config['alpha']  # Parameter alpha.
        self.model_ori = config['model_ori']
        self.train_stage = config.get('train', True)
        self.num_epochs_per_decay = config['num_epochs_per_decay']  # Epochs after which learning rate decays.
        self.initial_learning_rate_img = config['initial_learning_rate_img']  # Initial learning rate for image layer.
        self.save_dir = 'models/' + 'lr' + str(config['lr']) + '_epoch_' + str(config['max_epoch']) + '_' + str(config['codes_dim']) + 'bits'
        configProt = tf.ConfigProto()
        configProt.gpu_options.allow_growth = True
        configProt.allow_soft_placement = True
        self.sess = tf.Session(config=configProt)
        print('#######################################')
        print(self.save_dir)
        with tf.device(self.device):
            self.global_step = tf.Variable(dtype=tf.float32, initial_value=0, trainable=False, name='global_step_ddh')
            self.learning_rate = tf.Variable(dtype=tf.float32, initial_value=self.initial_learning_rate_img, trainable=False, name='lr_')
            self.imgs = tf.placeholder(tf.float32, [None, 256, 256, 3], name='input')
            self.label = tf.placeholder(tf.float32, [None, self.label_dim], name='label')
            self.codes = tf.placeholder(tf.float32, [None, self.codes_dim], name='codes')
            self.feature, self.bi = self.resnet(self.imgs)
            self.loss_functions()

    def d_p_function(self, dis, parameter):
        sim_cos = tf.exp(- parameter * dis)
        return sim_cos

    def kls_function(self, p, q, parameter):
        loss_1 = q * (p * tf.log((parameter * p) / ((parameter - 1) * p + 1)) + tf.log(parameter / (parameter - 1 + p)))
        loss_2 = (1 - q) * (p * tf.log(parameter / (parameter - 1)))
        loss = loss_1 + loss_2
        return loss

    def loss_functions(self):
        with tf.device(self.device):
            v_length = tf.sqrt(tf.reduce_sum(tf.square(self.feature), axis=1, keepdims=True))
            cos_1 = tf.matmul(self.feature, tf.transpose(self.feature))
            cos_2 = tf.matmul(v_length, tf.transpose(v_length))
            cos = tf.clip_by_value((1 - cos_1 / cos_2) * self.codes_dim * 0.5, 0.5, self.codes_dim)
            sim_cos = self.d_p_function(cos, self.lamda)
            sim_label = tf.clip_by_value(tf.matmul(self.label, tf.transpose(self.label)), 0.0, 1.0)
            kls = tf.reduce_mean(self.kls_function(sim_cos, sim_label, self.theta))
            ones = tf.ones(tf.shape(self.feature))
            cos_h_1 = tf.matmul(tf.abs(self.feature), tf.transpose(ones))
            v_1 = tf.sqrt(tf.reduce_sum(tf.square(self.feature), axis=1, keepdims=True))
            v_2 = tf.sqrt(tf.reduce_sum(ones, axis=1, keepdims=True))
            cos_h_2 = tf.matmul(v_1, tf.transpose(v_2))
            cos_h = (1 - cos_h_1 / cos_h_2) * self.codes_dim * 0.5
            q = tf.reduce_mean(self.d_p_function(cos_h, self.lamda))
            self.total_loss = kls + self.alpha * q

    def resnet(self, images):
        reshaped_image = tf.cast(images, tf.float32)
        distorted_image = tf.image.random_brightness(reshaped_image, max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
        distorted_image = tf.random_crop(distorted_image, [self.batch_size, 224, 224, 3], name='random_crop')
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3],
                           name='img_mean')
        distorted_image = distorted_image - mean
        output, net = resnet_v2.resnet_v2_50(distorted_image, num_classes=None, is_training=True)
        output = tf.squeeze(output, axis=[1, 2])
        with tf.name_scope('fc') as scope:
            fc3w = tf.Variable(tf.truncated_normal([2048, self.codes_dim],
                                                   dtype=tf.float32,
                                                   stddev=1e-2), name='weights')
            fc3b = tf.Variable(tf.constant(0.0, shape=[self.codes_dim], dtype=tf.float32), trainable=True,
                               name='biases')
            fc = tf.nn.bias_add(tf.matmul(output, fc3w), fc3b)
        feature = tf.nn.tanh(fc, name='feature')
        bi = tf.sign(feature, name='bi')
        return feature, bi

    def train_op(self):
        num_batches_per_epoch = int(self.n_train / self.batch_size)
        decay_steps = int(num_batches_per_epoch * self.num_epochs_per_decay * 2)
        var = tf.trainable_variables()
        var_cnn = tf.trainable_variables()[:len(var) - 2]
        var_fc = tf.trainable_variables()[len(var) - 2:]
        lr = tf.train.exponential_decay(self.learning_rate, self.global_step, decay_steps, 0.5, staircase=True)
        op_cnn = tf.train.MomentumOptimizer(learning_rate=lr / 10, momentum=0.9).minimize(
            self.total_loss, var_list=var_cnn, global_step=self.global_step)
        op_fc = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(
            self.total_loss, var_list=var_fc, global_step=self.global_step)
        apply_gradient_op = tf.group(op_cnn, op_fc)
        return apply_gradient_op

    def train(self, img_dataset):
        with tf.device(self.device):
            train_op = self.train_op()
            init = tf.global_variables_initializer()
            self.sess.run(init)
            variables_to_restore = slim.get_variables_to_restore()
            init_fn = slim.assign_from_checkpoint_fn(self.model_ori, variables_to_restore, ignore_missing_vars=True)
            init_fn(self.sess)

        for epoch in range(self.max_epoch):
            total_batch = int(self.n_train / self.batch_size)
            print("#Train# Start Epoch: " + str(epoch+1))
            print("#Train# training total batch is " + str(total_batch+1))
            epoch_loss, _per = 0, 0
            index = np.asarray([i for i in range(self.n_train)])
            np.random.shuffle(index)
            with tf.device(self.device):
                for i in range(total_batch):
                    images, image_labels = img_dataset.data(index[_per: _per+self.batch_size])
                    _per += self.batch_size
                    _, loss, bi = self.sess.run([train_op, self.total_loss, self.bi], feed_dict={self.imgs: images, self.label: image_labels})
                    epoch_loss += loss
                    assert not np.isnan(loss), 'Model diverged with loss = NaN'
                if self.n_train % self.batch_size != 0:
                    images, image_labels = img_dataset.data(index[-self.batch_size:])
                    _, loss, bi = self.sess.run([train_op, self.total_loss, self.bi],
                                                feed_dict={self.imgs: images, self.label: image_labels})
                    epoch_loss += loss
                    assert not np.isnan(loss), 'Model diverged with loss = NaN'
            epoch_loss = epoch_loss / total_batch
            print('##################################')
            print('epoch_loss=', epoch_loss)
        saver = tf.train.Saver()
        saver.save(self.sess, self.save_dir)


class validation(object):
    def __init__(self, dataset, query, config, model_trained):
        self.model_trained = model_trained
        self.model_trained_graph = self.model_trained + '.meta'
        self.re = dataset
        self.query = query
        self.device = config['device']
        self.label_dim = config['label_dim']
        self.codes_dim = config['codes_dim']
        self.batch_size = config['batch_size']
        self.n_train = config['n_train']
        self.n_database = config['n_database']
        self.n_query = config['n_query']
        self.recall = config['recall']
        return

    def get_mAP(self):
        configProt = tf.ConfigProto()
        configProt.gpu_options.allow_growth = True
        configProt.allow_soft_placement = True
        with tf.Session(config=configProt) as sess:
            with tf.device(self.device):
                saver = tf.train.import_meta_graph(self.model_trained_graph)
                saver.restore(sess, self.model_trained)
                total_batch = int(self.n_database / self.batch_size)
                print("######Validation###### Database total batch: " + str(total_batch + 1))
                graph = tf.get_default_graph()
                x, y = graph.get_tensor_by_name('input:0'), graph.get_tensor_by_name('bi:0')
                index = np.asarray([i for i in range(self.n_database)])
                re_codes, q_codes, re_labels, q_labels = np.zeros([self.n_database, self.codes_dim]), np.zeros(
                    [self.n_query, self.codes_dim]), np.zeros([self.n_database, self.label_dim]), np.zeros(
                    [self.n_query, self.label_dim])
                _per = 0
                for i in range(total_batch):
                    _re_images, _re_labels = self.re.data(index[_per: _per+self.batch_size])
                    _re_codes = sess.run([y], feed_dict={x: _re_images})
                    re_codes[_per: _per+self.batch_size, ...] = _re_codes[0]
                    re_labels[_per: _per+self.batch_size, ...] = _re_labels
                    _per += self.batch_size
                    print("# Dataset batch : ", str(i+1))
                if self.n_database % self.batch_size != 0:
                    _re_images, _re_labels = self.re.data(index[-self.batch_size:])
                    _re_codes = sess.run([y], feed_dict={x: _re_images})
                    re_codes[_per:, ...] = _re_codes[0][-(self.n_database % self.batch_size):, ...]
                    re_labels[_per:, ...] = _re_labels[-(self.n_database % self.batch_size):, ...]
                query_batch = int(self.n_query / self.batch_size)
                print("######Validation###### Query total batch: " + str(query_batch + 1))
                _per = 0
                index = np.asarray([i for i in range(self.n_query)])
                for i in range(query_batch):
                    _q_images, _q_labels = self.query.data(index[_per: _per+self.batch_size])
                    _q_codes = sess.run([y], feed_dict={x: _q_images})
                    q_codes[_per: _per+self.batch_size, ...] = _q_codes[0]
                    q_labels[_per: _per + self.batch_size, ...] = _q_labels
                    _per += self.batch_size
                    print("# Query batch : ", str(i + 1))
                if self.n_query % self.batch_size != 0:
                    _q_images, _q_labels = self.query.data(index[-self.batch_size:])
                    _q_codes = sess.run([y], feed_dict={x: _q_images})
                    q_codes[_per:, ...] = _q_codes[0][-(self.n_query % self.batch_size):, ...]
                    q_labels[_per:, ...] = _q_labels[-(self.n_query % self.batch_size):, ...]
                maps = evl.get_map(self.recall, re_codes, re_labels, q_codes, q_labels)
                return maps
