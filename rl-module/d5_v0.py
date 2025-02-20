'''
MIT License
Copyright (c) Chen-Yu Yen - Soheil Abbasloo 2020

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

import threading
import logging
import tensorflow as tf
import sys
from agent_v2 import Agent
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import gym
import numpy as np
import time
import random
import datetime
import sysv_ipc
import signal
import pickle
<<<<<<< HEAD:rl-module/d5_v0.py
from utils_v2 import logger, Params
from envwrapper_v2 import Env_Wrapper, TCP_Env_Wrapper, GYM_Env_Wrapper

=======
from utils import logger, Params
from envwrapper import Env_Wrapper, TCP_Env_Wrapper, GYM_Env_Wrapper
from constrained_reward import (
    parse_experiment_id,
    get_raw_and_constraint_reward,
)
>>>>>>> 7649c065ac68d242c2a2f10bec3f8a4e2bce87d8:rl-module/d5.py

def create_input_op_shape(obs, tensor):
    input_shape = [x or -1 for x in tensor.shape.as_list()]
    return np.reshape(obs, input_shape)


def initialize_symbolic_parameters(
    params,
    threshold,
):
    small_s1_delta = np.zeros([params.dict['state_dim']])
    small_s1_delta[-2] = threshold / 2.0
    small_s1_delta_c = threshold / 2.0
    large_s1_delta = np.zeros([params.dict['state_dim']])
    large_s1_delta[-2] = (1 - threshold) / 2.0
    large_s1_delta_c = (1 + threshold) / 2.0
    return small_s1_delta, small_s1_delta_c, large_s1_delta, large_s1_delta_c

def update_symbolic_parameters_from_s1(
    s1_rec_buffer, 
    params,
    small_s1_delta,
    small_s1_delta_c,
    large_s1_delta,
    large_s1_delta_c,
):
    small_s1_delta_rec_buffer = np.concatenate((np.zeros_like(s1_rec_buffer[params.dict['state_dim']:]), small_s1_delta))
    small_s1_c_rec_buffer = s1_rec_buffer
    small_s1_c_rec_buffer[-2] = small_s1_delta_c
    large_s1_delta_rec_buffer = np.concatenate((np.zeros_like(s1_rec_buffer[params.dict['state_dim']:]), large_s1_delta))
    large_s1_c_rec_buffer = s1_rec_buffer
    large_s1_c_rec_buffer[-2] = large_s1_delta_c
    return small_s1_delta_rec_buffer, small_s1_c_rec_buffer, large_s1_delta_rec_buffer, large_s1_c_rec_buffer


def evaluate_TCP(env, agent, epoch, summary_writer,
                 params, s0_rec_buffer, eval_step_counter,
                 training_log_f,
                 reward_mode,
                 constraints_id,
                 threshold,
                 evaluation_state_log_f,
                 x1,
                 x2,
                 lambda_
                 ):
    # Evaluate the TCP performance.
    '''
    training_log_f: The file to save training log.
    reward_mode: string, represent the reward computation mode.
    constraints_id: specify the property constraint.
    threshold: threshold for constraint satisfaction.
    evaluation_state_log_f: The file to save the state information.
    x1: the first changing point for the reward
    x2: the second changing point for the reward
    lambda_: the weight for the reward and the alpha_delta_reward
    '''

    score_list_raw_reward = []
    score_list_empirical_constraint_reward = []
    score_list_symbolic_constraint_reward = []
    score_list_overall_reward = []

    eval_times = 3
    eval_length = params.dict['max_eps_steps']
    start_time = time.time()

    key_property_idx = -2 # The index of inverseRTT in the state vector.
    # Create the s1_delta (only add the symbolic part to the last state[-2] (inverseRTT)) based on the threshold
    # When inverseRTT is small
    # Split the state into two parts: inverseRTT \in [0.0, threshold] and inverseRTT \in [threshold, 1.0].
    small_s1_delta, small_s1_delta_c, large_s1_delta, large_s1_delta_c = initialize_symbolic_parameters(
        params,
        threshold,
    )

    for _ in range(eval_times):

        step_counter = 0
        ep_raw_reward = 0.0
        ep_empirical_constraint_reward = 0.0
        ep_symbolic_constraint_reward = 0.0
        ep_overall_reward = 0.0

        if not params.dict['use_TCP']:
<<<<<<< HEAD:rl-module/d5_v0.py
            # TODO: move back to v0
            # s0 = env.reset()
            info = env.reset()
            s0 = info['state']
        
        # TODO: move back to v0
        # if params.dict['recurrent']:
        #     a = agent.get_action(s0_rec_buffer, False)
        # else:
        #     a = agent.get_action(s0, False)
=======
            info = env.reset()
            s0 = info['state']

>>>>>>> 7649c065ac68d242c2a2f10bec3f8a4e2bce87d8:rl-module/d5.py
        if params.dict['recurrent']:
            a = agent.get_concrete_action(s0_rec_buffer, False)
        else:
            a = agent.get_concrete_action(s0, False)
        a = a[0][0]
        previous_action = None
        previous_cwnd_tcp = None

        small_inverseRTT_a1_lower = None
        small_inverseRTT_a1_upper = None
        large_inverseRTT_a1_lower = None
        large_inverseRTT_a1_upper = None

        env.write_action(a)

        while True:

            eval_step_counter += 1
            step_counter += 1

<<<<<<< HEAD:rl-module/d5_v0.py
            # TODO: move back to v0
            # s1, r, terminal, error_code = env.step(a, eval_=True)
            info, r, terminal, error_code = env.step(a, eval_=True) 
            s1 = info['state']
=======
            info, r, terminal, error_code = env.step(a, eval_=True)
            s1 = info['state']
            cwnd_tcp = info['cwnd_tcp']
            evaluation_state_log_f.write(f"Epoch: {epoch}; state: {s1.tolist()}; action: {a}; cwnd_tcp: {cwnd_tcp}\n")
            raw_reward, empirical_constraint_reward, symbolic_constraint_reward, overall_reward = get_raw_and_constraint_reward(
                state=s1,
                action=a,
                raw_reward=r,
                current_tcp_cwnd=cwnd_tcp,
                previous_action=previous_action,
                previous_tcp_cwnd=previous_cwnd_tcp,
                constraints_id=constraints_id,
                reward_mode=reward_mode,
                threshold=threshold,
                x1=x1,
                x2=x2,
                lambda_=lambda_,
                small_inverseRTT_a1_lower=small_inverseRTT_a1_lower,
                small_inverseRTT_a1_upper=small_inverseRTT_a1_upper,
                large_inverseRTT_a1_lower=large_inverseRTT_a1_lower,
                large_inverseRTT_a1_upper=large_inverseRTT_a1_upper,
            )
            previous_action = a
            previous_cwnd_tcp = cwnd_tcp
>>>>>>> 7649c065ac68d242c2a2f10bec3f8a4e2bce87d8:rl-module/d5.py

            if error_code == True:
                s1_rec_buffer = np.concatenate((s0_rec_buffer[params.dict['state_dim']:], s1))
                small_s1_delta_rec_buffer, small_s1_c_rec_buffer, large_s1_delta_rec_buffer, large_s1_c_rec_buffer = update_symbolic_parameters_from_s1(
                    s0_rec_buffer, 
                    params,
                    small_s1_delta,
                    small_s1_delta_c,
                    large_s1_delta,
                    large_s1_delta_c,
                )

                # TODO: move back to v0
                # if params.dict['recurrent']:
                #     a1 = agent.get_action(s1_rec_buffer, False)
                # else:
                #     a1 = agent.get_action(s1, False)
                if params.dict['recurrent']:
                    a1 = agent.get_concrete_action(s1_rec_buffer, False)
<<<<<<< HEAD:rl-module/d5_v0.py
                else:
                    a1 = agent.get_concrete_action(s1, False)
=======
                    small_inverseRTT_a1_lower, small_inverseRTT_a1_upper = agent.get_symbolic_action(
                        s = small_s1_c_rec_buffer,
                        s_delta = small_s1_delta_rec_buffer,
                        use_noise = False,
                        )
                    large_inverseRTT_a1_lower, large_inverseRTT_a1_upper = agent.get_symbolic_action(
                        s = large_s1_c_rec_buffer,
                        s_delta = large_s1_delta_rec_buffer,
                        use_noise = False,
                        )
                    if agent.use_original or agent.use_snt_model_wo_ibp:
                        small_inverseRTT_a1_lower = small_inverseRTT_a1_lower[0][0][0]
                        small_inverseRTT_a1_upper = small_inverseRTT_a1_upper[0][0][0]
                        large_inverseRTT_a1_lower = large_inverseRTT_a1_lower[0][0][0]
                        large_inverseRTT_a1_upper = large_inverseRTT_a1_upper[0][0][0]
                else:
                    a1 = agent.get_concrete_action(s1, False)
                    raise ImportError("The recurrent mode is not supported in the evaluation.")
>>>>>>> 7649c065ac68d242c2a2f10bec3f8a4e2bce87d8:rl-module/d5.py

                a1 = a1[0][0]

                env.write_action(a1)

            else:
                print("Invalid state received...\n")
                env.write_action(a)
                continue
            
            ep_raw_reward += raw_reward
            ep_empirical_constraint_reward += empirical_constraint_reward
            ep_symbolic_constraint_reward += symbolic_constraint_reward
            ep_overall_reward += overall_reward

            if (step_counter+1) % params.dict['tb_interval'] == 0:
                summary = tf.summary.Summary()
                summary.value.add(tag='Eval/Step/0-Actions', simple_value=env.map_action(a))
                summary.value.add(tag='Eval/Step/2-RawReward', simple_value=raw_reward)
                summary.value.add(tag='Eval/Step/2-EmpiricalConstraintReward', simple_value=empirical_constraint_reward)
                summary.value.add(tag='Eval/Step/2-SymbolicConstraintReward', simple_value=symbolic_constraint_reward)
                summary.value.add(tag='Eval/Step/2-OverallReward', simple_value=overall_reward)
            summary_writer.add_summary(summary, eval_step_counter)

            s0 = s1
            a = a1
            if params.dict['recurrent']:
                s0_rec_buffer = s1_rec_buffer

            if step_counter == eval_length or terminal:
                score_list_raw_reward.append(ep_raw_reward)
                score_list_empirical_constraint_reward.append(ep_empirical_constraint_reward)
                score_list_symbolic_constraint_reward.append(ep_symbolic_constraint_reward)
                score_list_overall_reward.append(ep_overall_reward)
                break

    summary = tf.summary.Summary()
    summary.value.add(tag='Eval/Return/RawReward', simple_value=np.mean(score_list_raw_reward))
    summary.value.add(tag='Eval/Return/EmpiricalConstraintReward', simple_value=np.mean(score_list_empirical_constraint_reward))
    summary.value.add(tag='Eval/Return/SymbolicConstraintReward', simple_value=np.mean(score_list_symbolic_constraint_reward))
    summary.value.add(tag='Eval/Return/OverallReward', simple_value=np.mean(score_list_overall_reward))
    summary_writer.add_summary(summary, epoch)
<<<<<<< HEAD:rl-module/d5_v0.py
    print(f"Epoch: {epoch}; Eval/Return: {np.mean(score_list)}")
=======
    print(f"Epoch: {epoch}, Eval/Return/RawReward: {np.mean(score_list_raw_reward)}")
    print(f"Epoch: {epoch}, Eval/Return/EmpiricalConstraintReward: {np.mean(score_list_empirical_constraint_reward)}")
    print(f"Epoch: {epoch}, Eval/Return/SymbolicConstraintReward: {np.mean(score_list_symbolic_constraint_reward)}")
    print(f"Epoch: {epoch}, Eval/Return/OverallReward: {np.mean(score_list_overall_reward)}")
    training_log_f.write(f"Epoch: {epoch}, Eval/Return/RawReward: {np.mean(score_list_raw_reward)}; ")
    training_log_f.write(f"Eval/Return/EmpiricalConstraintReward: {np.mean(score_list_empirical_constraint_reward)}; ")
    training_log_f.write(f"Eval/Return/SymbolicConstraintReward: {np.mean(score_list_symbolic_constraint_reward)}; ")
    training_log_f.write(f"Eval/Return/OverallReward: {np.mean(score_list_overall_reward)}\n")
    training_log_f.flush()
    evaluation_state_log_f.flush()
>>>>>>> 7649c065ac68d242c2a2f10bec3f8a4e2bce87d8:rl-module/d5.py

    return eval_step_counter


class learner_killer():

    def __init__(self, buffer):
        self.replay_buf = buffer
        print("learner register sigterm")
        signal.signal(signal.SIGTERM, self.handler_term)
        print("test length:", self.replay_buf.length_buf)

    def handler_term(self, signum, frame):
        # TODO: To fix! Now saving replay buffer sometimes makes the sequential experiments stuck. Comment them out for now.
        # if not config.eval:
        #     with open(os.path.join(params.dict['train_dir'], "replay_memory.pkl"), "wb") as fp:
        #         pickle.dump(self.replay_buf, fp)
        #         print("test length:", self.replay_buf.length_buf)
        #         print("--------------------------Learner: Saving rp memory--------------------------")
        print("-----------------------Learner's killed---------------------")
        sys.exit(0)


def main():

    tf.get_logger().setLevel(logging.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true', default=False, help='default is  %(default)s')
    parser.add_argument('--eval', action='store_true', default=False, help='default is  %(default)s')
    parser.add_argument('--tb_interval', type=int, default=1)
    parser.add_argument('--train_dir', type=str, default=None)
    parser.add_argument('--mem_r', type=int, default = 123456)
    parser.add_argument('--mem_w', type=int, default = 12345)
    parser.add_argument('--base_path',type=str, required=True)
    parser.add_argument('--job_name', type=str, choices=['learner', 'actor'], required=True, help='Job name: either {\'learner\', actor}')
    parser.add_argument('--task', type=int, required=True, help='Task id')
    parser.add_argument('--experiment_id', type=str, required=True, help='Experiment id')
    parser.add_argument('--constraints_id', type=int, required=True, help='Constraints id')
    parser.add_argument('--max_actor_epochs', type=int, default=50000, help='Set the upper bound of actor epochs')
    parser.add_argument('--threshold', type=float, required=True, help='threshold hold for the constraint satisfaction')
    parser.add_argument('--x1', type=float, default=5, help='the first changing point for the reward')
    parser.add_argument('--x2', type=float, default=25, help='the second changing point for the reward')
    parser.add_argument('--lambda_', type=float, default=0.5, help='the weight for the reward and the alpha_delta_reward')
    parser.add_argument('--original_model', type=int, default=1, help='Use the original model or not')
    parser.add_argument('--snt_model_wo_ibp', type=int, default=0, help='Use the snt model without IBP or not')

    ## parameters from parser
    global config
    global params
    config = parser.parse_args()
    reward_mode = parse_experiment_id(config.experiment_id)
    # TODO: Modify the following part in the scripts later.
    constraints_id = config.constraints_id
    max_actor_epochs = config.max_actor_epochs
    threshold = config.threshold
    x1 = config.x1
    x2 = config.x2
    lambda_ = config.lambda_
    use_original_model = config.original_model == 1
    use_snt_model_wo_ibp = config.snt_model_wo_ibp == 1

    print(f"-----------job name: {config.job_name}, threshold: {threshold}, x1: {x1}, x2: {x2}, lambda_: {lambda_}, use_original_model: {use_original_model}, use_snt_model_wo_ibp: {use_snt_model_wo_ibp}-----------")

    # parameters from file
    params = Params(os.path.join(config.base_path,'params.json'))

    if params.dict['single_actor_eval']:
        local_job_device = ''
        shared_job_device = ''
        def is_actor_fn(i): return True
        global_variable_device = '/cpu'
        is_learner = False
        server = tf.train.Server.create_local_server()
        filters = []
    else:
        local_job_device = '/job:%s/task:%d' % (config.job_name, config.task)
        shared_job_device = '/job:learner/task:0'

        is_learner = config.job_name == 'learner'
        global_variable_device = shared_job_device + '/cpu'

        def is_actor_fn(i): return config.job_name == 'actor' and i == config.task

        if params.dict['remote']:
            cluster = tf.train.ClusterSpec({
                'actor': params.dict['actor_ip'][:params.dict['num_actors']],
                'learner': [params.dict['learner_ip']]
            })
        else:
            cluster = tf.train.ClusterSpec({
                    'actor': ['localhost:%d' % (8001 + i) for i in range(params.dict['num_actors'])],
                    'learner': ['localhost:8000']
                })

        server = tf.train.Server(cluster, job_name=config.job_name,
                                task_index=config.task)
        filters = [shared_job_device, local_job_device]

    if params.dict['use_TCP']:
        env_str = "TCP"
        env_peek = TCP_Env_Wrapper(env_str, params,use_normalizer=params.dict['use_normalizer'])

    else:
        env_str = 'YourEnvironment'
        env_peek =  Env_Wrapper(env_str)

    s_dim, a_dim = env_peek.get_dims_info()
    action_scale, action_range = env_peek.get_action_info()

    if not params.dict['use_TCP']:
        params.dict['state_dim'] = s_dim
    if params.dict['recurrent']:
        s_dim = s_dim * params.dict['rec_dim']

    if params.dict['use_hard_target'] == True:
        params.dict['tau'] = 1.0

    with tf.Graph().as_default(),\
        tf.device(local_job_device + '/cpu'):

        tf.set_random_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        actor_op = []
        now = datetime.datetime.now()
        tfeventdir = os.path.join(config.base_path, params.dict['logdir'],
                                  config.job_name+str(config.task)+'-'+str(config.experiment_id)+'-'+str(constraints_id)+'-'+str(threshold)+'-'+str(max_actor_epochs))
        params.dict['train_dir'] = tfeventdir
        if config.job_name == 'actor':
            if config.eval:
                log_directory = 'evaluation_log'
            else:
                log_directory = 'training_log' 
            experiment_name_suffix = config.job_name+str(config.task)+'-'+str(config.experiment_id)+'-'+str(constraints_id)+'-'+str(threshold)+'-'+str(max_actor_epochs)+'-'+str(x1)+'-'+str(x2)+'-'+str(lambda_)
            training_log_path = os.path.join(config.base_path, log_directory,
                                            f"{experiment_name_suffix}.txt")
            evaluation_state_log_path = os.path.join(config.base_path, log_directory,
                                            f"{experiment_name_suffix+'-EvalStates'}.txt")
            training_state_log_path = os.path.join(config.base_path, log_directory,
                                            f"{experiment_name_suffix+'-TrainStates'}.txt")
            training_log_f = open(training_log_path, 'w')
            evaluation_state_log_f = open(evaluation_state_log_path, 'w')
            training_state_log_f = open(training_state_log_path, 'w')

        if not os.path.exists(tfeventdir):
            os.makedirs(tfeventdir)
        summary_writer = tf.summary.FileWriterCache.get(tfeventdir)

        with tf.device(shared_job_device):
            agent = Agent(s_dim, a_dim, batch_size=params.dict['batch_size'], summary=summary_writer,h1_shape=params.dict['h1_shape'],
                        h2_shape=params.dict['h2_shape'],stddev=params.dict['stddev'],mem_size=params.dict['memsize'],gamma=params.dict['gamma'],
                        lr_c=params.dict['lr_c'],lr_a=params.dict['lr_a'],tau=params.dict['tau'],PER=params.dict['PER'],CDQ=params.dict['CDQ'],
                        LOSS_TYPE=params.dict['LOSS_TYPE'],noise_type=params.dict['noise_type'],noise_exp=params.dict['noise_exp'],
                        use_original=use_original_model, use_snt_model_wo_ibp=use_snt_model_wo_ibp)

            dtypes = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
            shapes = [[s_dim], [a_dim], [1], [s_dim], [1]]
            queue = tf.FIFOQueue(10000, dtypes, shapes, shared_name="rp_buf")


        if is_learner:
            with tf.device(params.dict['device']):
                agent.build_learn()

                agent.create_tf_summary()

            if config.load is True and config.eval==False:
                if os.path.isfile(os.path.join(params.dict['train_dir'], "replay_memory.pkl")):
                    with open(os.path.join(params.dict['train_dir'], "replay_memory.pkl"), 'rb') as fp:
                        replay_memory = pickle.load(fp)

            _killsignal = learner_killer(agent.rp_buffer)


        for i in range(params.dict['num_actors']):
                if is_actor_fn(i):
                    if params.dict['use_TCP']:
                        shrmem_r = sysv_ipc.SharedMemory(config.mem_r)
                        shrmem_w = sysv_ipc.SharedMemory(config.mem_w)
                        env = TCP_Env_Wrapper(env_str, params, config=config, for_init_only=False, shrmem_r=shrmem_r, shrmem_w=shrmem_w,use_normalizer=params.dict['use_normalizer'])
                    else:
                        env = GYM_Env_Wrapper(env_str, params)

                    a_s0 = tf.placeholder(tf.float32, shape=[s_dim], name='a_s0')
                    a_action = tf.placeholder(tf.float32, shape=[a_dim], name='a_action')
                    a_reward = tf.placeholder(tf.float32, shape=[1], name='a_reward')
                    a_s1 = tf.placeholder(tf.float32, shape=[s_dim], name='a_s1')
                    a_terminal = tf.placeholder(tf.float32, shape=[1], name='a_terminal')
                    a_buf = [a_s0, a_action, a_reward, a_s1, a_terminal]

                    with tf.device(shared_job_device):
                        actor_op.append(queue.enqueue(a_buf))

        if is_learner:
            Dequeue_Length = params.dict['dequeue_length']
            dequeue = queue.dequeue_many(Dequeue_Length)

        queuesize_op = queue.size()

        if params.dict['ckptdir'] is not None:
            params.dict['ckptdir'] = os.path.join( config.base_path, params.dict['ckptdir'])
            print("## checkpoint dir:", params.dict['ckptdir'])
            isckpt = os.path.isfile(os.path.join(params.dict['ckptdir'], 'checkpoint') )
            print("## checkpoint exists?:", isckpt)
            if isckpt== False:
                print("\n# # # # # # Warning ! ! ! No checkpoint is loaded, use random model! ! ! # # # # # #\n")
        else:
            params.dict['ckptdir'] = tfeventdir

        tfconfig = tf.ConfigProto(allow_soft_placement=True)

        if params.dict['single_actor_eval']:
            mon_sess = tf.train.SingularMonitoredSession(
                checkpoint_dir=params.dict['ckptdir'])
        else:
            mon_sess = tf.train.MonitoredTrainingSession(
                    master=server.target,
                    save_checkpoint_secs=15,
                    save_summaries_secs=None,
                    save_summaries_steps=None,
                    is_chief=is_learner,
                    checkpoint_dir=params.dict['ckptdir'],
                    config=tfconfig,
                    hooks=None)

        agent.assign_sess(mon_sess)

        if is_learner:

            if config.eval is True:
                print("=========================Learner is up===================")
                while not mon_sess.should_stop():
                    time.sleep(1)
                    continue

            if config.load is False:
                agent.init_target()

            counter = 0
            start = time.time()

            dequeue_thread = threading.Thread(target=learner_dequeue_thread, args=(agent,params, mon_sess, dequeue, queuesize_op, Dequeue_Length),daemon=True)
            first_time=True

            while not mon_sess.should_stop():

                if first_time == True:
                    dequeue_thread.start()
                    first_time=False

                up_del_tmp=params.dict['update_delay']/1000.0
                time.sleep(up_del_tmp)
                if agent.rp_buffer.ptr>200 or agent.rp_buffer.full:
                    agent.train_step()
                    if params.dict['use_hard_target'] == False:
                        agent.target_update()
                        if counter %params.dict['hard_target'] == 0 :
                            current_opt_step = agent.sess.run(agent.global_step)
                            # TODO: To revisit the logger printing here.
                    else:
                        if counter %params.dict['hard_target'] == 0 :
                            agent.target_update()
                            current_opt_step = agent.sess.run(agent.global_step)
                            logger.info("Optimize step:{}".format(current_opt_step))
                            logger.info("rp_buffer ptr:{}".format(agent.rp_buffer.ptr))
                counter += 1

        else:
<<<<<<< HEAD:rl-module/d5_v0.py
                start = time.time()
                step_counter = np.int64(0)
                eval_step_counter = np.int64(0)
                # TODO: move back to v0
                # s0 = env.reset()
                info = env.reset()
                s0 = info['state']
                s0_rec_buffer = np.zeros([s_dim])
                s1_rec_buffer = np.zeros([s_dim])
                s0_rec_buffer[-1*params.dict['state_dim']:] = s0

                # TODO: move back to v0
                # if params.dict['recurrent']:
                #     a = agent.get_action(s0_rec_buffer,not config.eval)
                # else:
                #     a = agent.get_action(s0, not config.eval)
                if params.dict['recurrent']:
                    a = agent.get_concrete_action(s0_rec_buffer, not config.eval)
                else:
                    a = agent.get_concrete_action(s0, not config.eval)
                a = a[0][0]
                env.write_action(a)
                epoch = 0
                ep_r = 0.0
                start = time.time()
                while True:
                    start = time.time()
                    epoch += 1

                    step_counter += 1
                    # TODO: move back to v0
                    # s1, r, terminal, error_code = env.step(a,eval_=config.eval)
                    info, r, terminal, error_code = env.step(a,eval_=config.eval)
                    s1 = info['state']

                    if error_code == True:
                        s1_rec_buffer = np.concatenate((s0_rec_buffer[params.dict['state_dim']:], s1))

                        # TODO: move back to v0
                        # if params.dict['recurrent']:
                        #     a1 = agent.get_action(s1_rec_buffer, not config.eval)
                        # else:
                        #     a1 = agent.get_action(s1,not config.eval)
                        if params.dict['recurrent']:
                            a1 = agent.get_concrete_action(s1_rec_buffer, not config.eval)
                        else:
                            a1 = agent.get_concrete_action(s1, not config.eval)
=======
            start = time.time()
>>>>>>> 7649c065ac68d242c2a2f10bec3f8a4e2bce87d8:rl-module/d5.py

            small_s1_delta, small_s1_delta_c, large_s1_delta, large_s1_delta_c = initialize_symbolic_parameters(
                params,
                threshold,
            )

            small_inverseRTT_a1_lower = None
            small_inverseRTT_a1_upper = None
            large_inverseRTT_a1_lower = None
            large_inverseRTT_a1_upper = None

            step_counter = np.int64(0)
            eval_step_counter = np.int64(0)
            info = env.reset()
            s0 = info['state']
            s0_rec_buffer = np.zeros([s_dim])
            s1_rec_buffer = np.zeros([s_dim])
            s0_rec_buffer[-1*params.dict['state_dim']:] = s0

            if params.dict['recurrent']:
                a = agent.get_concrete_action(s0_rec_buffer, not config.eval)
            else:
                a = agent.get_concrete_action(s0, not config.eval)
            a = a[0][0]
            env.write_action(a)
            previous_action = None
            previous_cwnd_tcp = None
            epoch = 0
            start = time.time()
            # while True:
            while epoch < max_actor_epochs:
                start = time.time()
                epoch += 1

                step_counter += 1
                info, r, terminal, error_code = env.step(a, eval_=config.eval)
                s1 = info['state']
                cwnd_tcp = info['cwnd_tcp']
                raw_reward, empirical_constraint_reward, symbolic_constraint_reward, overall_reward = get_raw_and_constraint_reward(
                    state=s1,
                    action=a,
                    raw_reward=r,
                    current_tcp_cwnd=cwnd_tcp,
                    previous_action=previous_action,
                    previous_tcp_cwnd=previous_cwnd_tcp,
                    constraints_id=constraints_id,
                    reward_mode=reward_mode,
                    threshold=threshold,
                    x1=x1,
                    x2=x2,
                    lambda_=lambda_,
                    small_inverseRTT_a1_lower=small_inverseRTT_a1_lower,
                    small_inverseRTT_a1_upper=small_inverseRTT_a1_upper,
                    large_inverseRTT_a1_lower=large_inverseRTT_a1_lower,
                    large_inverseRTT_a1_upper=large_inverseRTT_a1_upper,
                )
                previous_action = a
                previous_cwnd_tcp = cwnd_tcp
                training_state_log_f.write(f"Epoch: {epoch}; state: {s1.tolist()}; action: {a}; cwnd_tcp: {cwnd_tcp}\n")
                training_state_log_f.flush()

                if error_code == True:
                    s1_rec_buffer = np.concatenate( (s0_rec_buffer[params.dict['state_dim']:], s1) )

                    small_s1_delta_rec_buffer, small_s1_c_rec_buffer, large_s1_delta_rec_buffer, large_s1_c_rec_buffer = update_symbolic_parameters_from_s1(
                        s1_rec_buffer, 
                        params,
                        small_s1_delta,
                        small_s1_delta_c,
                        large_s1_delta,
                        large_s1_delta_c,
                    )

                    if params.dict['recurrent']:
                        a1 = agent.get_concrete_action(s1_rec_buffer, not config.eval)
                        small_inverseRTT_a1_lower, small_inverseRTT_a1_upper = agent.get_symbolic_action(
                            s=small_s1_c_rec_buffer,
                            s_delta=small_s1_delta_rec_buffer,
                            use_noise=False,
                            )
                        large_inverseRTT_a1_lower, large_inverseRTT_a1_upper = agent.get_symbolic_action(
                            s=large_s1_c_rec_buffer,
                            s_delta=large_s1_delta_rec_buffer,
                            use_noise=False,
                            )
                        if agent.use_original or agent.use_snt_model_wo_ibp:
                            small_inverseRTT_a1_lower = small_inverseRTT_a1_lower[0][0][0]
                            small_inverseRTT_a1_upper = small_inverseRTT_a1_upper[0][0][0]
                            large_inverseRTT_a1_lower = large_inverseRTT_a1_lower[0][0][0]
                            large_inverseRTT_a1_upper = large_inverseRTT_a1_upper[0][0][0]
                    else:
                        a1 = agent.get_concrete_action(s1, not config.eval)
                        raise ImportError("The recurrent mode is not supported in the evaluation.")
                    a1 = a1[0][0]
                    env.write_action(a1)
                else:
                    print("TaskID:"+str(config.task)+"Invalid state received...\n")
                    env.write_action(a)
                    continue
                
                if agent.use_original or agent.use_snt_model_wo_ibp:
                    assigned_action = a
                else:
                    assigned_action = [a]
                if params.dict['recurrent']:
                    fd = {a_s0:s0_rec_buffer, a_action:np.array(assigned_action), a_reward:np.array([overall_reward]), a_s1:s1_rec_buffer, a_terminal:np.array([terminal], np.float)}
                else:
                    fd = {a_s0:s0, a_action:np.array(assigned_action), a_reward:np.array([overall_reward]), a_s1:s1, a_terminal:np.array([terminal], np.float)}

                if not config.eval:
                    mon_sess.run(actor_op, feed_dict=fd)

                s0 = s1
                a = a1
                if params.dict['recurrent']:
                    s0_rec_buffer = s1_rec_buffer

                if not params.dict['use_TCP'] and (terminal):
                    if agent.actor_noise != None:
                        agent.actor_noise.reset()

                if (epoch% params.dict['eval_frequency'] == 0):
                    eval_step_counter = evaluate_TCP(env, agent, epoch, summary_writer,
                                                    params, s0_rec_buffer, eval_step_counter, 
                                                    training_log_f, reward_mode, constraints_id,
                                                    threshold, evaluation_state_log_f,
                                                    x1=x1,
                                                    x2=x2,
                                                    lambda_=lambda_,
                                                    )

            print("total time:", time.time()-start)

def learner_dequeue_thread(agent,params, mon_sess, dequeue, queuesize_op, Dequeue_Length):
    ct = 0
    while True:
        ct = ct + 1
        data = mon_sess.run(dequeue)
        agent.store_many_experience(data[0], data[1], data[2], data[3], data[4], Dequeue_Length)
        time.sleep(0.01)


def learner_update_thread(agent,params):
    delay=params.dict['update_delay']/1000.0
    ct = 0
    while True:
        agent.train_step()
        agent.target_update()
        time.sleep(delay)


if __name__ == "__main__":
    main()
