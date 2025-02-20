import os
import tensorflow as tf
import interval_bound_propagation as ibp
import sonnet as snt
import numpy as np
from sklearn.model_selection import train_test_split
from agent_v2 import extract_predictor_output_bound
import warnings

def create_input_op_shape(obs, tensor):
    input_shape = [x or -1 for x in tensor.shape.as_list()]
    return np.reshape(obs, input_shape)


# Define the neural transition model
class NN(snt.AbstractModule):
    def __init__(self, s_dim, a_dim, h1_shape, h2_shape, latency_scale=1.0, name="nn_transition"):
        super(NN, self).__init__(name=name)
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.name = name
        self.latency_scale = latency_scale
        self.h1_shape = h1_shape
        self.h2_shape = h2_shape
    
    def train_var(self):
        # return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    
    def global_var(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
    
    def _build(self, s, is_training):
        '''
        We need to use snt layer structures to make ibp usable for this tf module.
        '''
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            m = snt.Linear(output_size=self.h1_shape, use_bias=True, name='fc1')
            h1 = m(s)
            # Use the ibp wrapper directly.
            # m = ibp.BatchNorm(name='bn1')
            m = snt.BatchNorm(name='bn1')
            h1 = m(h1, is_training=is_training)
            h1 = tf.nn.leaky_relu(h1)

            m = snt.Linear(output_size=self.h2_shape, use_bias=True, name='fc2')
            h2 = m(h1)
            # Use the ibp wrapper directly.
            # m = ibp.BatchNorm(name='bn2')
            m = snt.BatchNorm(name='bn2')
            h2 = m(h2, is_training=is_training)
            h2 = tf.nn.leaky_relu(h2)
            
            m = snt.Linear(output_size=self.a_dim, use_bias=True, name='fc3')
            h3 = m(h2)
            output = tf.nn.tanh(h3)
            output = tf.add(output, 1.0)
            scale_output = tf.multiply(output, self.latency_scale) 

        return scale_output
    

class NeuralTransition():
    def __init__(self, o_dim=8, a_dim=1, h1_shape=256, h2_shape=256, latency_scale=0.5, mode='single'):
        tf.reset_default_graph()
        
        self.mode = mode
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.h1_shape = h1_shape
        self.h2_shape = h2_shape
        self.latency_scale = latency_scale
        self.learning_rate = 0.001
        self.global_step = tf.train.get_or_create_global_step(graph=None)
        
        if self.mode == 'single':
            assert(self.o_dim == 8)
        elif self.mode == 'recurrent':
            assert(self.o_dim == 80)
        elif self.mode == 'recurrent2':
            assert(self.o_dim == 16)
        else:
            raise ValueError("mode should be either 'single' or 'recurrent'")
        
        self.transition_directory = "/proj/VerifiedMLSys/ConstrainedOrca"
        self.trace_data_dir = f"{self.transition_directory}/multi_bw_trace_dataset"
        self.train_dir = f"{self.transition_directory}/transition_{mode}_model"
        
        self.x = tf.placeholder(tf.float32, [None, self.o_dim], name='x')
        self.x_delta = tf.placeholder(tf.float32, shape=(None, self.o_dim), name="sa_delta")
        self.y = tf.placeholder(tf.float32, shape=(None), name="inverseRTT")
        self.is_training = tf.placeholder(tf.bool, name='Transition_is_training')
        self.is_concrete = tf.placeholder(tf.bool, name='Transition_is_concrete')
        
        self.transition = NN(self.o_dim, self.a_dim, self.h1_shape, self.h2_shape, self.latency_scale)
        # self.concrete_y_pred = self.transition._build(self.x, is_training=self.is_training)
        concrete_y_pred = self.transition(self.x, is_training=self.is_training)
        model_wrapper = ibp.crown.VerifiableModelWrapper
        self.symbolic_y_predictor = model_wrapper(self.transition)
        
        self.concrete_y_pred = self.symbolic_y_predictor(self.x, is_training=self.is_training)
        self.symbolic_y_pred_lower, self.symbolic_y_pred_upper = extract_predictor_output_bound(
            self.symbolic_y_predictor,
            self.x, 
            self.is_concrete, 
            self.x_delta,
            )
        # TODO: Test with changing the self.concrete_y_pred to symbolic_y_pred_lower with delta==0
        self.mse = tf.reduce_mean(tf.square(self.y - self.concrete_y_pred), name='mse')
        self.rmse = tf.sqrt(self.mse, name='rmse')
        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.training_op = self.optimizer.minimize(self.rmse, var_list=self.transition.train_var())
        
        self.saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=5)
    
    def assign_sess(self, sess):
        self.sess = sess
        
    def load_dataset(self):
        all_trace_files = os.listdir(self.trace_data_dir)
        X_data, y_data = [], []
        for trace_file in all_trace_files:
            if trace_file.endswith('.txt'):
                single_trace_X_data, single_trace_y_data = self.load_multi_bw_trace(
                    os.path.join(self.trace_data_dir, trace_file)
                    )
                X_data.extend(single_trace_X_data)
                y_data.extend(single_trace_y_data)
                # TODO:
                break
        X = np.array(X_data)
        y = np.array(y_data)
        return X, y
    
    def get_symbolic_action(self, o, o_delta):
        # Really get the symbolic action lower bound and upper bound from the IBP wrapped model.
        fd = {self.x: create_input_op_shape(o, self.x),
                self.is_training:False,
                self.is_concrete:False,
                self.x_delta: create_input_op_shape(o_delta, self.x_delta)
                }
        # TODO: 
        # latency_lower_bound, latency_upper_bound = self.sess.run(
        #     [self.symbolic_y_pred_lower, self.symbolic_y_pred_upper],
        #     feed_dict=fd,
        #     )
        latency_lower_bound = self.sess.run(self.symbolic_y_pred_lower, feed_dict=fd)
        latency_upper_bound = self.sess.run(self.symbolic_y_pred_upper, feed_dict=fd)
        return latency_lower_bound, latency_upper_bound

    def get_concrete_action(self, o):
        # Get the concrete action from the model.
        fd = {self.x: create_input_op_shape(o, self.x),
                self.is_training:False,
                self.is_concrete:True,
                self.x_delta: create_input_op_shape(np.zeros_like(o), self.x_delta)
                }
        # TODO:
        # latency_lower_bound, latency_upper_bound = self.sess.run(
        #     [self.symbolic_y_pred_lower, self.symbolic_y_pred_upper],
        #     feed_dict=fd,
        # )
        latency_lower_bound = self.sess.run(self.symbolic_y_pred_lower, feed_dict=fd)
        latency_upper_bound = self.sess.run(self.symbolic_y_pred_upper, feed_dict=fd)
        # TODO:
        if latency_lower_bound != latency_upper_bound:
            warnings.warn(f"latency_lower_bound {latency_lower_bound} != latency_upper_bound {latency_upper_bound}")
            assert(latency_lower_bound == latency_upper_bound)
        assert(latency_lower_bound == latency_upper_bound)
        return latency_lower_bound
        
    def train_save(self, epoch_num, dataset_size=None):
        # TODO:
        # 1. load the dataset
        X_data, y_data = self.load_dataset()
        # TODO
        if dataset_size is None:
            X_train, X_test, y_train, y_test = train_test_split(
                X_data, y_data, test_size=0.05, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_data[:dataset_size], y_data[:dataset_size], test_size=0.05, random_state=42)
        
        batch_size = 256
    
        def iteration(x, y, bs=128):
            for i in range(0, len(x), bs):
                yield x[i:i+bs], y[i:i+bs]
            return
        
        epoch_num = epoch_num
        
        for i in range(epoch_num):
            for x_batch_train, y_batch_train in iteration(X_train, y_train, bs=batch_size):
                self.sess.run(self.training_op, feed_dict={self.x: x_batch_train, 
                                                           self.y: y_batch_train,
                                                           self.is_training: True,
                                                           self.is_concrete: True
                                                           })
            if i % 1000 == 0:
                rmse_list = []
                for x_batch_test, y_batch_test in iteration(X_test, y_test, bs=batch_size):
                    rmse_val = self.rmse.eval(session=self.sess, 
                                              feed_dict={self.x: x_batch_test,
                                                         self.y: y_batch_test,
                                                         self.is_training: False,
                                                         self.is_concrete: True
                                                         })
                    rmse_list.append(rmse_val)
                print("Epoch: ", i, "RMSE: ", np.mean(rmse_list))
                print(f"tmp_latency: {self.get_concrete_action(o=[[0]*self.o_dim],)}")
                print(f"tmp_latency: {self.get_concrete_action(o=[[0.5]*self.o_dim],)}")
                print(f"tmp_latency: {self.get_concrete_action(o=[[1]*self.o_dim],)}")
                latency_lower_1, latency_upper_1 = self.get_symbolic_action(
                    o=[[1]*self.o_dim],
                    o_delta=[[0]*self.o_dim],
                )
                print(f"latency: {latency_lower_1}, {latency_upper_1}")
                latency_lower_2, latency_upper_2 = self.get_symbolic_action(
                    o=[[1]*self.o_dim],
                    o_delta=[[0.5]*self.o_dim],
                )
                print(f"latency: {latency_lower_2}, {latency_upper_2}")
                self.saver.save(self.sess, f'{self.train_dir}/test_model_{self.mode}', global_step=i)
        print("Model saved in path: %s" % f'{self.train_dir}/test_model_{self.mode}')
    
    def load_model(self, epoch_num):
        # tf.global_variables_initializer().run(session=self.sess)
        # TODO: add maximum epoch here.
        self.saver = tf.train.import_meta_graph(f'{self.train_dir}/test_model_{self.mode}-{epoch_num}.meta')
        # Initialize the variables (may not be trainable.)
        self.sess.run(tf.global_variables_initializer())
        # Load all the important variables and overwrite the values from global_variables_initializer()
        self.saver.restore(self.sess, tf.train.latest_checkpoint(f'{self.train_dir}/'))
        
    def load_multi_bw_trace(self, trace_file):
        with open(trace_file, "r") as f:
            lines = f.readlines()
            full_dataset = []
            X_data = []
            y_data = []
            for line in lines:
                trace = line[:-1].split(",")
                trace_numbers = [float(x) for x in trace]
                full_dataset.extend(trace_numbers)
            full_dataset_size = len(full_dataset)
            # if mode is 'single', the dataset is previous state (7) + cwnd + next state [-2]
            for i in range(0, full_dataset_size-self.o_dim, self.o_dim):
                if i + self.o_dim + 5 > full_dataset_size:
                    break
                # the past state + the current cwnd
                X_data.append(full_dataset[i:i+self.o_dim])
                # the next latency
                y_data.append(full_dataset[i+self.o_dim+5])
        return X_data, y_data
        