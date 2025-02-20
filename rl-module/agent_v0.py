'''
  MIT License
  Copyright (c) Chen-Yu Yen 2020

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
'''


import tensorflow as tf
import interval_bound_propagation as ibp
import sonnet as snt
import numpy as np
import os
import time
import warnings

EXPLORE = 4000
STDDEV = 0.1
NSTEP = 0.3


from utils_v0 import OU_Noise, ReplayBuffer, G_Noise, Prioritized_ReplayBuffer

def create_input_op_shape(obs, tensor):
    input_shape = [x or -1 for x in tensor.shape.as_list()]
    return np.reshape(obs, input_shape)

class Actor():
    def __init__(self, s_dim, a_dim,h1_shape,h2_shape, action_scale=1.0, name='actor'):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.name = name
        self.action_scale = action_scale
        self.h1_shape = h1_shape
        self.h2_shape = h2_shape

    def train_var(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def build(self, s, is_training):

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            h1 = tf.layers.dense(s, units=self.h1_shape, name='fc1')
            h1 = tf.layers.batch_normalization(h1, training=is_training, scale=False)
            h1 = tf.nn.leaky_relu(h1)

            h2 = tf.layers.dense(h1, units=self.h2_shape, name='fc2')
            h2 = tf.layers.batch_normalization(h2, training=is_training, scale=False)
            h2 = tf.nn.leaky_relu(h2)

            output = tf.layers.dense(h2, units=self.a_dim, activation=tf.nn.tanh)

            scale_output = tf.multiply(output, self.action_scale)

        return scale_output


class AbstractActor(snt.AbstractModule):
  def __init__(self, s_dim, a_dim, h1_shape, h2_shape, action_scale=1.0, name='abstract_actor'):
    super(AbstractActor, self).__init__(name=name)
    self.s_dim = s_dim
    self.a_dim = a_dim
    self.name = name
    self.action_scale = action_scale
    self.h1_shape = h1_shape
    self.h2_shape = h2_shape

  def train_var(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

  def _build(self, s, is_training):
    '''
    We need to use snt layer structures to make ibp usable for this tf module.
    '''
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      m = snt.Linear(output_size=self.h1_shape, use_bias=True, name='fc1')
      h1 = m(s)
      # m = ibp.BatchNorm(name='bn1')
      m = snt.BatchNorm(name='bn1')
      h1 = m(h1, is_training=is_training)
      h1 = tf.nn.leaky_relu(h1)

      m = snt.Linear(output_size=self.h2_shape, use_bias=True, name='fc2')
      h2 = m(h1)
      # m = ibp.BatchNorm(name='bn2')
      m = snt.BatchNorm(name='bn2')
      h2 = m(h2, is_training=is_training)
      h2 = tf.nn.leaky_relu(h2)

      m = snt.Linear(output_size=self.a_dim, use_bias=True, name='fc1')
      h3 = m(h2)
      output = tf.nn.tanh(h3)
      scale_output = tf.multiply(output, self.action_scale) 

    return scale_output 


class Critic():
    def __init__(self, s_dim, a_dim,h1_shape,h2_shape, action_scale=1.0, name='critic'):
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.name = name
        self.action_scale = action_scale
        self.h1_shape = h1_shape
        self.h2_shape = h2_shape

    def train_var(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def build(self, s, action):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):

            h1 = tf.layers.dense(s, units=self.h1_shape, activation=tf.nn.leaky_relu, name='fc1')

            h2 = tf.layers.dense(tf.concat([h1, action], -1), units=self.h2_shape, activation=tf.nn.leaky_relu, name='fc2')
            output = tf.layers.dense(h2, units=1)

        return output


def extract_predictor_output_bound(predictor, s0, is_concrete=True, s0_delta=None):
    '''
    Extract bounds from the predictor output.
    
    Arguments:
    - predictor: a tf module that has been trained.
    - s0: the input state.
    - is_concrete: whether to use concrete IBP, where the delta should be all zero.
    - s0_delta: the perturbation to the input state. If is_concrete is True, this should be all zero.
    
    Returns:
    - lower_bound: the lower bound of the output. (if is_concrete is True, this is the concrete lower bound)
    - upper_bound: the upper bound of the output.
    '''
    # TODO: check the input shapes.
    delta = tf.cond(is_concrete, lambda: tf.zeros_like(s0), lambda: s0_delta)
    input_interval_bounds = ibp.IntervalBounds(s0 - delta, s0 + delta)
    # Set up the bound propagation.
    predictor.propagate_bounds(input_interval_bounds)
    # Get the output bounds.
    modules = predictor.modules
    print("modules")
    print(modules)
    bounds = modules[-1].output_bounds
    output_bound_upper = bounds._upper
    output_bound_lower = bounds._lower

    return output_bound_lower, output_bound_upper


class Agent():
    def __init__(self, s_dim, a_dim, h1_shape,h2_shape,gamma=0.995, batch_size=8, lr_a=1e-4, lr_c=1e-3, tau=1e-3, mem_size=1e5,action_scale=1.0, action_range=(-1.0, 1.0),
<<<<<<< HEAD:rl-module/agent_v0.py
                noise_type=3, noise_exp=50000, summary=None,stddev=0.1, PER=False, alpha=0.6, CDQ=True, LOSS_TYPE='HUBERT'):
        print("--------Agent v0--------")
=======
                noise_type=3, noise_exp=50000, summary=None,stddev=0.1, PER=False, alpha=0.6, CDQ=True, LOSS_TYPE='HUBERT',
                use_original=True, use_snt_model_wo_ibp=False):
        ####### For debug #######
        self.use_original = use_original
        self.use_snt_model_wo_ibp = use_snt_model_wo_ibp
        # use_original: Use the original actor model for concrete action and symbolic action.
        # use_snt_model_wo_ibp: Use the snt actor model for concrete action and symbolic action, but without the IBP wrapper.
        # ~use_original and ~use_snt_model_wo_ibp: Use the snt actor model with the IBP wrapper for symbolic action, and set the delta to 0.0 for concrete action.
        ######################### 
        print("use original", self.use_original)
        print("use snt model without ibp", self.use_snt_model_wo_ibp)
        
>>>>>>> 7649c065ac68d242c2a2f10bec3f8a4e2bce87d8:rl-module/agent.py
        self.PER = PER
        self.CDQ = CDQ
        self.LOSS_TYPE = LOSS_TYPE
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.gamma = gamma
        self.train_dir = './train_dir'
        self.step_epochs = tf.Variable(0, trainable=False, name='epoch')
        self.global_step = tf.train.get_or_create_global_step(graph=None)

        self.s0 = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='s0')
        self.s0_delta = tf.placeholder(tf.float32, shape=[None, s_dim], name='s0_delta')
        self.s1 = tf.placeholder(tf.float32, shape=[None, self.s_dim], name='s1')
        self.is_training = tf.placeholder(tf.bool, name='Actor_is_training')
        self.is_concrete = tf.placeholder(tf.bool, name='Actor_is_concrete')
        self.action = tf.placeholder(tf.float32, shape=[None, a_dim], name='action')
        self.noise_type = noise_type
        self.noise_exp = noise_exp
        self.action_range = action_range
        self.h1_shape=h1_shape
        self.h2_shape=h2_shape
        self.stddev=stddev
        if not self.PER:
            self.rp_buffer = ReplayBuffer(int(mem_size), s_dim, a_dim, batch_size=batch_size)
        else:
            self.rp_buffer = Prioritized_ReplayBuffer(int(mem_size), s_dim, a_dim, batch_size=batch_size, alpha=alpha)

        if noise_type == 1:
            self.actor_noise = OU_Noise(mu=np.zeros(a_dim), sigma=float(self.stddev) * np.ones(a_dim),dt=1,exp=self.noise_exp)
        elif noise_type == 2:
            ## Gaussian with gradually decay
            self.actor_noise = G_Noise(mu=np.zeros(a_dim), sigma=float(self.stddev) * np.ones(a_dim), explore =self.noise_exp)
        elif noise_type == 3:
            ## Gaussian without gradually decay
            self.actor_noise = G_Noise(mu=np.zeros(a_dim), sigma=float(self.stddev) * np.ones(a_dim), explore = None,theta=0.1)
        elif noise_type == 4:
            ## Gaussian without gradually decay
            self.actor_noise = G_Noise(mu=np.zeros(a_dim), sigma=float(self.stddev) * np.ones(a_dim), explore = EXPLORE,theta=0.1,mode="step",step=NSTEP)
        elif noise_type == 5:
            self.actor_noise = None
        else:
            self.actor_noise = OU_Noise(mu=np.zeros(a_dim), sigma=float(self.stddev) * np.ones(a_dim),dt=0.5)

        # Main Actor/Critic Network
        self.original_actor = Actor(self.s_dim, self.a_dim, action_scale=action_scale, h1_shape=self.h1_shape, h2_shape=self.h2_shape)
        self.symbolic_actor = AbstractActor(self.s_dim, self.a_dim, action_scale=action_scale, h1_shape=self.h1_shape, h2_shape=self.h2_shape)

        if not self.use_original and not self.use_snt_model_wo_ibp:
            concrete_action = self.symbolic_actor(self.s0, is_training=self.is_training)
            model_wrapper = ibp.crown.VerifiableModelWrapper
            self.symbolic_actor_perdictor = model_wrapper(self.symbolic_actor)

        self.critic = Critic(self.s_dim, self.a_dim, action_scale=action_scale, h1_shape=self.h1_shape, h2_shape=self.h2_shape)
        self.critic2 = Critic(self.s_dim, self.a_dim, action_scale=action_scale, name='critic2', h1_shape=self.h1_shape, h2_shape=self.h2_shape)
        
        self.original_actor_out = self.original_actor.build(self.s0, self.is_training) # Add the epsilon here.

        if not self.use_original and not self.use_snt_model_wo_ibp:
            self.concrete_actor_out = self.symbolic_actor_perdictor(self.s0, is_training=self.is_training)
            self.symbolic_actor_bound_lower, self.symbolic_actor_bound_upper = extract_predictor_output_bound(self.symbolic_actor_perdictor, self.s0, self.is_concrete, self.s0_delta)
        elif self.use_snt_model_wo_ibp:
            self.concrete_actor_out = self.symbolic_actor._build(self.s0, self.is_training)
        self.critic_out = self.critic.build(self.s0, self.action)
        self.critic_out2 = self.critic2.build(self.s0, self.action)

        if self.use_original:
            self.critic_actor_out = self.critic.build(self.s0, self.original_actor_out) # set is_concrete to True
        elif self.use_snt_model_wo_ibp:
            self.critic_actor_out = self.critic.build(self.s0, self.concrete_actor_out)
        else:
            self.critic_actor_out = self.critic.build(self.s0, self.symbolic_actor_bound_lower) # set is_concrete to True

        # Target Actor/Critic network
        if self.use_original:
            self.target_actor = Actor(self.s_dim, self.a_dim, action_scale=action_scale, h1_shape=self.h1_shape, h2_shape=self.h2_shape, name="target_actor")
        else:
            self.target_actor = AbstractActor(self.s_dim, self.a_dim, action_scale=action_scale, h1_shape=self.h1_shape, h2_shape=self.h2_shape, name="target_actor")
        
        self.target_critic = Critic(self.s_dim, self.a_dim, action_scale=action_scale ,h1_shape=self.h1_shape,h2_shape=self.h2_shape,name='target_critic')
        self.target_critic2 = Critic(self.s_dim, self.a_dim, action_scale=action_scale, name='target_critic2',h1_shape=self.h1_shape,h2_shape=self.h2_shape)

        if self.use_original:
            self.target_actor_out = self.target_actor.build(self.s1, self.is_training)
        else:
            self.target_actor_out = self.target_actor._build(self.s1, self.is_training)
        self.target_actor_policy = self.get_target_actor_policy()
        self.target_critic_actor_out = self.target_critic.build(self.s1, self.target_actor_policy)
        self.target_critic_actor_out2 = self.target_critic2.build(self.s1, self.target_actor_policy)

        if self.use_original:
            self.target_actor_update_op = self.target_update_op(self.target_actor.train_var(), self.original_actor.train_var(), tau)
        else:
            self.target_actor_update_op = self.target_update_op(self.target_actor.train_var(), self.symbolic_actor.train_var(), tau)
        self.target_critic_update_op = self.target_update_op(self.target_critic.train_var(), self.critic.train_var(), tau)
        self.target_critic_update_op2 = self.target_update_op(self.target_critic2.train_var(), self.critic2.train_var(), tau)

        if self.use_original:
            self.target_act_init_op = self.target_init(self.target_actor.train_var(), self.original_actor.train_var())
        else:
            self.target_act_init_op = self.target_init(self.target_actor.train_var(), self.symbolic_actor.train_var())
        self.target_cri_init_op = self.target_init(self.target_critic.train_var(), self.critic.train_var())
        self.target_cri_init_op2 = self.target_init(self.target_critic2.train_var(), self.critic2.train_var())

        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        self.terminal = tf.placeholder(tf.float32, shape=[None, 1], name='is_terminal')
        self.reward = tf.placeholder(tf.float32, shape=[None, 1], name='reward')
        self.y = self.reward + self.gamma * (1-self.terminal) * self.target_critic_actor_out
        self.y2 = self.reward + self.gamma * (1-self.terminal) * self.target_critic_actor_out2

        self.importance = tf.placeholder(tf.float32, [None,1], name='imporance_weights')
        self.td_error = self.critic_out - self.y

        self.summary_writer = summary

    def build_learn(self):
        self.actor_optimizer = tf.train.AdamOptimizer(self.lr_a)
        self.critic_optimizer = tf.train.AdamOptimizer(self.lr_c)

        self.actor_train_op = self.build_actor_train_op()

        use_huber = True

        if use_huber:
            self.critic_train_op = self.build_critic_train_op_huber()
        else:
            self.critic_train_op = self.build_critic_train_op()

    def build_critic_train_op_huber(self):

        def f1(y,pred, weights=1.0):
            error = tf.square(y - pred)
            weighted_error = tf.reduce_mean(error*weights)
            return weighted_error

        loss_function = {
            'HUBER':tf.compat.v1.losses.huber_loss,
            'MSE': f1
        }
        if self.CDQ:
            q_min_target = tf.minimum(self.y, self.y2)

            if self.PER:
                self.c_loss = loss_function[self.LOSS_TYPE](q_min_target, self.critic_out, weights=self.importance)
                self.c_loss2 = loss_function[self.LOSS_TYPE](q_min_target, self.critic_out2, weights=self.importance)

            else:
                self.c_loss = loss_function[self.LOSS_TYPE](q_min_target, self.critic_out)
                self.c_loss2 = loss_function[self.LOSS_TYPE](q_min_target, self.critic_out2)

            critic_op = []
            critic_op.append(self.critic_optimizer.minimize(self.c_loss, var_list=self.critic.train_var(), global_step = self.global_step))
            critic_op.append(self.critic_optimizer.minimize(self.c_loss2, var_list=self.critic2.train_var(), global_step = self.global_step))
            return critic_op
        else:
            if self.PER:
                self.critic_loss = loss_function[self.LOSS_TYPE](self.y, self.critic_out, weights=self.importance)
            else:
                self.critic_loss = loss_function[self.LOSS_TYPE](self.y, self.critic_out)
            loss_op = self.critic_optimizer.minimize(self.critic_loss, var_list=self.critic.train_var(), global_step = self.global_step)
            return loss_op

    def create_tf_summary(self):
        if self.CDQ:
            tf.summary.scalar('Loss/critic_loss:', self.c_loss)
            tf.summary.scalar('Loss/critic_loss_2:', self.c_loss2)
        else:
            tf.summary.scalar('Loss/critic_loss:', self.critic_loss)

        tf.summary.scalar('Loss/actor_loss:', self.a_loss)

        self.summary_op = tf.summary.merge_all()

    def init_target(self):
        self.sess.run(self.target_act_init_op)
        self.sess.run(self.target_cri_init_op)
        self.sess.run(self.target_cri_init_op2)

    def get_target_actor_policy(self):
        eps = tf.random_normal(tf.shape(self.target_actor_out), stddev=0.1)
        eps = tf.clip_by_value(eps, -0.2, 0.2)
        t_a = self.target_actor_out + eps
        t_a = tf.clip_by_value(t_a, -1.0, 1.0)
        return t_a

    def assign_sess(self, sess):
        self.sess = sess

    def build_critic_train_op(self):
        critic_op = []
        _q_min_target = tf.minimum(self.y, self.y2)

        q_min_target = _q_min_target
        self.c_loss = tf.reduce_mean(tf.square(q_min_target - self.critic_out))
        self.c_loss2 = tf.reduce_mean(tf.square(q_min_target - self.critic_out2))
        critic_op.append(self.critic_optimizer.minimize(self.c_loss, var_list=self.critic.train_var(), global_step = self.global_step))
        critic_op.append(self.critic_optimizer.minimize(self.c_loss2, var_list=self.critic2.train_var(), global_step = self.global_step))
        tf.summary.scalar('Loss/critic_loss:', self.c_loss)
        tf.summary.scalar('Loss/critic_loss_2:', self.c_loss2)
        return critic_op

    def build_actor_train_op(self):
        self.a_loss = -tf.reduce_mean(self.critic_actor_out)
        if self.use_original:
            return self.actor_optimizer.minimize(self.a_loss, var_list=self.original_actor.train_var(), global_step = self.global_step)
        else:
            return self.actor_optimizer.minimize(self.a_loss, var_list=self.symbolic_actor.train_var(), global_step = self.global_step)

    def target_init(self, target, vars):
        return [tf.assign(target[i], vars[i]) for i in range(len(vars))]

    def target_update_op(self, target, vars, tau):
        return [tf.assign(target[i], vars[i] * tau + target[i] * (1 - tau)) for i in range(len(vars))]

    def target_update_hard_op(self, target, vars):
        return [tf.assign(target[i], vars[i]) for i in range(len(vars))]

    def target_update(self):
        self.sess.run([self.target_actor_update_op, self.target_critic_update_op, self.target_critic_update_op2])

    def actor_clone_update(self):
        self.sess.run(self.actor_clone_update_op)

<<<<<<< HEAD:rl-module/agent_v0.py
    # TODO: move back to v0
    # def get_action(self, s, use_noise=True):
    def get_concrete_action(self, s, use_noise=True):

        fd = {self.s0: create_input_op_shape(s, self.s0), self.is_training:False}

        action = self.sess.run([self.actor_out], feed_dict=fd)
        if use_noise:
            noise = self.actor_noise(action[0])
            action += noise
            action = np.clip(action, self.action_range[0], self.action_range[1])
        return action

=======
    def get_concrete_action(self, s, use_noise=True):
        if self.use_original:
            # Use the original Actor model for concrete action.
            fd = {self.s0: create_input_op_shape(s, self.s0), self.is_training:False}
            action = self.sess.run([self.original_actor_out], feed_dict=fd)
            if use_noise:
                noise = self.actor_noise(action[0])
                action += noise
                action = np.clip(action, self.action_range[0], self.action_range[1])
            return action
        elif self.use_snt_model_wo_ibp:
            # Use the snt Actor model without the IBP wrapper for concrete action.
            fd = {self.s0: create_input_op_shape(s, self.s0), self.is_training:False}
            action = self.sess.run([self.concrete_actor_out], feed_dict=fd)
            if use_noise:
                noise = self.actor_noise(action[0])
                action += noise
                action = np.clip(action, self.action_range[0], self.action_range[1])
            return action
        else:
            # Use the snt Actor model + IBP wrapper. Get the action by setting the delta as 0.0.
            fd = {self.s0: create_input_op_shape(s, self.s0),
                self.is_training:False,
                self.is_concrete:True,
                self.s0_delta: create_input_op_shape(np.zeros_like(s), self.s0_delta)
                }
            action_lower_bound, action_upper_bound = self.sess.run([self.symbolic_actor_bound_lower, self.symbolic_actor_bound_upper], feed_dict=fd)
            # Make sure the action lower bound and upper bound are the same given the zero delta.
            if action_lower_bound != action_upper_bound:
                warnings.warn(f"action_lower_bound {action_lower_bound} != action_upper_bound {action_upper_bound} with input s {s}.")
                assert(action_lower_bound == action_upper_bound)
            # Set the action as the action lower bound. Here, action_lower_bound and action_upper_bound are the concrete action. They are the same.
            action = action_lower_bound
            if use_noise:
                noise = self.actor_noise(action[0])
                action += noise
                action = np.clip(action, self.action_range[0], self.action_range[1])
            return action
    
    def get_symbolic_action(self, s, s_delta, use_noise=False):
        if self.use_original or self.use_snt_model_wo_ibp:
            # When using the concrete actor model (no IBP wrapper).
            action = self.get_concrete_action(s, use_noise=False)
            return action, action
        else:
            # Really get the symbolic action lower bound and upper bound from the IBP wrapped model.
            fd = {self.s0: create_input_op_shape(s, self.s0),
                    self.is_training:False,
                    self.is_concrete:False, 
                    self.s0_delta: create_input_op_shape(s_delta, self.s0_delta)
                    }
            action_lower_bound, action_upper_bound = self.sess.run([self.symbolic_actor_bound_lower, self.symbolic_actor_bound_upper], feed_dict=fd)
            # print(f"## Symbolic action -- action_lower_bound {action_lower_bound} & action_upper_bound {action_upper_bound} with input s {s}.")
            if use_noise:
                raise NotImplementedError(f"use_noise {use_noise} is not supported yet.")
            return action_lower_bound, action_upper_bound
    
>>>>>>> 7649c065ac68d242c2a2f10bec3f8a4e2bce87d8:rl-module/agent.py
    def get_q(self, s, a):
        fd = {self.s0: create_input_op_shape(s, self.s0),
              self.action: create_input_op_shape(a, self.action)}

        return self.sess.run([self.critic_out], feed_dict=fd)

    def get_q_actor(self, s):
        if self.use_original or self.use_snt_model_wo_ibp:
            fd = {
                self.s0: create_input_op_shape(s, self.s0),
                self.is_training:True,
                }
            return self.sess.run([self.critic_actor_out], feed_dict=fd)
        else:
            fd = {
                self.s0: create_input_op_shape(s, self.s0),
                self.is_training:True,
                self.is_concrete:True,
                self.s0_delta: create_input_op_shape(np.zeros_like(s), self.s0_delta)
                }
            # The critic_actor_out is the concrete value of the Q value.
            return self.sess.run([self.critic_actor_out], feed_dict=fd)

    def store_experience(self, s0, a, r, s1, terminal):
        self.rp_buffer.store(s0, a, r, s1, terminal)

    def store_many_experience(self, s0, a, r, s1, terminal, length):
        if self.PER:
            for i in range(length):
                self.rp_buffer.store(s0[i], a[i], r[i], s1[i], terminal[i])
        else:
            self.rp_buffer.store_many(s0, a, r, s1, terminal, length)

    def sample_experince(self):
        return self.rp_buffer.sample()

    def train_step_td(self):
        return None

    def train_step(self):
        extra_update_ops = [v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS) if
                            "actor" in v.name and "target" not in v.name]

        if self.PER == True:
            batch_samples, weights, idxes = self.rp_buffer.sample()

            t2 = time.time()
            fd = {self.s0: create_input_op_shape(batch_samples[0], self.s0),
                self.action: create_input_op_shape(batch_samples[1], self.action),
                self.reward: create_input_op_shape(batch_samples[2], self.reward),
                self.s1: create_input_op_shape(batch_samples[3], self.s1),
                self.terminal: create_input_op_shape(batch_samples[4], self.terminal),
                self.is_training: True,
                self.importance: np.expand_dims(weights,axis=1),
                self.is_concrete: True,
                self.s0_delta: create_input_op_shape(np.zeros_like(batch_samples[0]), self.s0_delta)
                }
        else:
            batch_samples = self.rp_buffer.sample()
            t2 = time.time()
            fd = {self.s0: create_input_op_shape(batch_samples[0], self.s0),
                self.action: create_input_op_shape(batch_samples[1], self.action),
                self.reward: create_input_op_shape(batch_samples[2], self.reward),
                self.s1: create_input_op_shape(batch_samples[3], self.s1),
                self.terminal: create_input_op_shape(batch_samples[4], self.terminal),
                self.is_training: True,
                self.is_concrete: True,
                self.s0_delta: create_input_op_shape(np.zeros_like(batch_samples[0]), self.s0_delta)
                }

        if self.PER:
            _, td_errors = self.sess.run([self.critic_train_op, self.td_error], feed_dict=fd)
        else:
            self.sess.run([self.critic_train_op], feed_dict=fd)

        self.sess.run([self.actor_train_op, extra_update_ops], feed_dict=fd)

        summary, step = self.sess.run([self.summary_op, self.global_step], feed_dict=fd)

        self.summary_writer.add_summary(summary, global_step=step)

        if self.PER:
            new_priorities = np.abs(np.squeeze(td_errors)) + 1e-6
            self.rp_buffer.update_priorities(idxes, new_priorities)

    def log_tf(self, val, tag=None, step_counter=0):
        summary = tf.Summary()
        summary.value.add(tag= tag, simple_value=val)
        self.summary_writer.add_summary(summary, step_counter)

    def save_model(self, step=None):
        self.saver.save(self.sess, os.path.join(self.train_dir, 'model'), global_step =step)

    def load_model(self, name=None):
        if name is not None:
            print(os.path.join(self.train_dir, name))
            self.saver.restore(self.sess, os.path.join(self.train_dir, name))
        else:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.train_dir))

    def updat_step_epochs(self, epoch):
        self.sess.run(tf.assign(self.step_epochs, epoch))

    def get_step_epochs(self):
        return self.sess.run(self.step_epochs)
