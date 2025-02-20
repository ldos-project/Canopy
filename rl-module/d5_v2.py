"""
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
"""

import json
import threading
import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import sys
from agent_v2 import Agent

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import argparse
import numpy as np
import time
import random
import datetime
import sysv_ipc
import signal
import pickle
from utils_v2 import logger, Params
from envwrapper_v2 import Env_Wrapper, TCP_Env_Wrapper, GYM_Env_Wrapper
from constrained_reward import (
    convert_reward_mode,
    get_raw_and_constraint_reward,
    SAFETY_CONSTRAINTS_ID,
    ROBUSTNESS_CONSTRAINTS_ID,
    LOSS_CONSTRAINTS_ID,
    LOSS_CONSTRAINTS_LIVENESS_ID,
    PERF_ROBUSTNESS_CONSTRAINTS_ID,
    DEEP_BUFFER_CONSTRAINTS,
    SHALLOW_BUFFER_CONSTRAINTS,
)
from symbolic_transitions import (
    initialize_symbolic_spec_single_step_only_latency,
    update_symbolic_s_single_step,
    get_symbolic_actions,
)
from constants import (
    SymbolicComponent_bound,
)
from utils_v2 import get_forced_cwnd_from_alpha_and_tcp_cwnd


def create_input_op_shape(obs, tensor):
    input_shape = [x or -1 for x in tensor.shape.as_list()]
    return np.reshape(obs, input_shape)


def initialize_symbolic_spec(
    symbolic_mode,
    threshold,
    k_symbolic_components,
    constraints_id,
):
    if (
        symbolic_mode == "single_step_only_latency"
        or symbolic_mode == "multi_step_only_latency"
    ):
        s1_symbolic = initialize_symbolic_spec_single_step_only_latency(
            threshold,
            k=k_symbolic_components,
            constraints_id=constraints_id,
        )
    else:
        raise ValueError(f"symbolic_mode {symbolic_mode} is not supported.")
    return s1_symbolic


def evaluate_TCP(
    env,
    agent,
    epoch,
    params,
    s0,
    s0_rec_buffer,
    s0_rec_buffer_symbolic,
    eval_step_counter,
    training_log_f,
    reward_mode,
    constraints_id,
    threshold=None,
    evaluation_state_log_f=None,
    x1=None,
    x2=None,
    lambda_=None,
    k_symbolic_components=None,
    k=None,
    evaluation_reward_and_state_log_f=None,
    symbolic_mode="single_step_only_latency",
    actor_id=0,
    eval=False,
    training_session_idx=-1,
    start_time=None,
    only_tcp=False,
):
    # Evaluate the TCP performance.
    """
    training_log_f: The file to save training log.
    reward_mode: string, represent the reward computation mode.
    constraints_id: specify the property constraint.
    threshold: threshold for constraint satisfaction.
    evaluation_state_log_f: The file to save the state information.
    x1: the first changing point for the reward
    x2: the second changing point for the reward
    lambda_: the weight for the reward and the alpha_delta_reward
    k_symbolic_components: the number of symbolic components.
    k: the number of steps to consider for the long horizon constraints.
    """
    # TODO: assign s0_rec_buffer_symbolic
    score_list_raw_reward = []
    score_list_empirical_constraint_reward = []
    score_list_symbolic_constraint_reward = []
    score_list_overall_reward = []

    eval_times = 1
    eval_length = params.dict["max_eps_steps"]

    # Create the s1_delta (only add the symbolic part to the last state[-2] (inverseRTT)) based on the threshold
    # When inverseRTT is small
    # Split the state into two parts: inverseRTT \in [0.0, threshold] and inverseRTT \in [threshold, 1.0].
    # This is for one step's state.
    initial_s1_symbolic = initialize_symbolic_spec(
        symbolic_mode,
        threshold,
        k_symbolic_components,
        constraints_id,
    )

    for _ in range(eval_times):

        step_counter = 0
        ep_raw_reward = 0.0
        ep_empirical_constraint_reward = 0.0
        ep_symbolic_constraint_reward = 0.0
        ep_overall_reward = 0.0

        if not params.dict["use_TCP"]:
            info = env.reset()
            s0 = info["state"]

        if only_tcp:
            a = 0
            evaluation_state_log_f = None
        else:
            if params.dict["recurrent"]:
                a, a_before_noise = agent.get_concrete_action(s0_rec_buffer, False)
                a_symbolic, union_a1_symbolic = get_symbolic_actions(
                    s0_rec_buffer_symbolic,
                    agent,
                    constraints_id,
                )
            else:
                a, a_before_noise = agent.get_concrete_action(s0, False)
                raise ValueError(
                    "Non-recurrent mode is not supported for symbolic actions yet."
                )
            a = a[0][0]
        past_states = []
        past_actions = []
        past_tcp_cwnds = []
        # for symbolic transitions
        past_cwnds = []

        if constraints_id == PERF_ROBUSTNESS_CONSTRAINTS_ID:
            # Perf + robustness
            union_a_symbolic = {
                'perf': SymbolicComponent_bound(constraints_id=SAFETY_CONSTRAINTS_ID),
                'robustness': SymbolicComponent_bound(constraints_id=ROBUSTNESS_CONSTRAINTS_ID),
            }
        else:
            union_a_symbolic = SymbolicComponent_bound(
                constraints_id,
                None,
                None,
                None,
                None,
                None,
                None,
            )
        a_symbolic = None

        env.write_action(a)

        while True:
            eval_step_counter += 1
            step_counter += 1

            # Get s', cwnd_tcp' from a
            info, r, terminal, error_code = env.step(a, eval_=True)
            s1 = info["state"]
            cwnd_tcp = info["cwnd_tcp"]  # Attention! This is the cwnd_tcp with a.
            cwnd = get_forced_cwnd_from_alpha_and_tcp_cwnd(a, cwnd_tcp)

            # s, a, s'
            # concrete reward is over (s, a, s')
            # symbolic/heuristic rewards are over (s, a); if s, s_t-1, ...: a xxx.
            if only_tcp:
                raw_reward = r
                symbolic_constraint_reward = 0.0
                empirical_constraint_reward = 0.0
                overall_reward = r
            else:
                (
                    raw_reward,
                    empirical_constraint_reward,
                    symbolic_constraint_reward,
                    overall_reward,
                    symbolic_cwnd_info,
                    detailed_symbolic_cwnd_info,
                    heuristic_constraints_info,
                ) = get_raw_and_constraint_reward(
                    state=s0,
                    action=a,
                    action_before_noise=a_before_noise,
                    raw_reward=r,
                    current_tcp_cwnd=cwnd_tcp,
                    constraints_id=constraints_id,
                    reward_mode=reward_mode,
                    threshold=threshold,
                    x1=x1,
                    x2=x2,
                    lambda_=lambda_,
                    union_a_symbolic=union_a_symbolic,
                    past_states=past_states,  # not including the current state (effect from the current action)
                    past_actions=past_actions,  # not including the current action
                    past_tcp_cwnds=past_tcp_cwnds,  # not including the current cwnd_tcp (mapping with the current action)
                    k=k,
                    a_symbolic=a_symbolic,
                )


            if evaluation_reward_and_state_log_f:
                evaluation_reward_and_state_log_f.write(
                    f"Time: {time.time()}; states: {s0.tolist()}; action: {a}; cwnd_tcp: {cwnd_tcp}; raw_reward: {raw_reward}; certified_reward: {symbolic_constraint_reward}; combined_reward: {overall_reward}; action: {a}\n"
                )
                evaluation_reward_and_state_log_f.flush()
            if evaluation_state_log_f:
                evaluation_state_log_f.write(
                    f"Time: {time.time()}; Epoch: {epoch}; state: {s1.tolist()}; action: {a}; cwnd_tcp: {cwnd_tcp};"
                )
                if constraints_id in {
                        SAFETY_CONSTRAINTS_ID,
                        LOSS_CONSTRAINTS_ID,
                        LOSS_CONSTRAINTS_LIVENESS_ID,
                        PERF_ROBUSTNESS_CONSTRAINTS_ID,
                        DEEP_BUFFER_CONSTRAINTS,
                        SHALLOW_BUFFER_CONSTRAINTS,
                    }:
                    evaluation_state_log_f.write(
                        f" small_inverseRTT_cwnd: [{detailed_symbolic_cwnd_info['small_inverseRTT_delta_cwnd_lower']}, {detailed_symbolic_cwnd_info['small_inverseRTT_delta_cwnd_upper']}];"
                    )
                    evaluation_state_log_f.write(
                        f" large_inverseRTT_cwnd: [{detailed_symbolic_cwnd_info['large_inverseRTT_delta_cwnd_lower']}, {detailed_symbolic_cwnd_info['large_inverseRTT_delta_cwnd_upper']}];"
                    )
                if constraints_id in {
                        ROBUSTNESS_CONSTRAINTS_ID,
                        PERF_ROBUSTNESS_CONSTRAINTS_ID,
                    }:
                    evaluation_state_log_f.write(
                        f" cwnd: [{detailed_symbolic_cwnd_info['l']}, {detailed_symbolic_cwnd_info['u']}];"
                    )
                evaluation_state_log_f.write(
                    f" latency_signal: {heuristic_constraints_info['latency_signal']}, delta_cwnd: {heuristic_constraints_info['delta_cwnd']}\n"
                )
                evaluation_state_log_f.flush()


            ep_raw_reward += raw_reward
            ep_empirical_constraint_reward += empirical_constraint_reward
            ep_symbolic_constraint_reward += symbolic_constraint_reward
            ep_overall_reward += overall_reward

            # from s, a, s', a'
            if error_code == True:
                # By default, we use recurrent mode.
                s1_rec_buffer = np.concatenate(
                    (s0_rec_buffer[params.dict["state_dim"] :], s1)
                )
                if not only_tcp:
                    # if k > 1 and len(past_states) >= 10 + k:
                    #     # TODO: Below are for multi-step symbolic transitions.
                    #     raise ValueError(
                    #         "Multi-step symbolic transition is not implemented yet."
                    #     )
                    # else:  # single step condition
                    s1_rec_buffer_symbolic = update_symbolic_s_single_step(
                        s1_rec_buffer,
                        initial_s1_symbolic,
                        constraints_id,
                    )
                    a1, a1_before_noise = agent.get_concrete_action(
                        s1_rec_buffer, False
                    )
                    a1_symbolic, union_a1_symbolic = get_symbolic_actions(
                        s1_rec_buffer_symbolic, agent, constraints_id
                    )
                    a1 = a1[0][0]
                else:
                    a1 = 0
                env.write_action(a1)
            else:
                print("Invalid state received...\n")
                env.write_action(a)
                continue

            past_actions.append(a)
            past_states.append(s0)
            past_tcp_cwnds.append(cwnd_tcp)
            past_cwnds.append(cwnd)

            if (
                len(past_states) > k + 10
            ):  # TODO: Don't bother too long history for now.
                past_states.pop(0)
                past_actions.pop(0)
                past_tcp_cwnds.pop(0)
                past_cwnds.pop(0)

            if not only_tcp:
                a_symbolic = a1_symbolic
                union_a_symbolic = union_a1_symbolic

                if not only_tcp:
                    if params.dict["recurrent"]:
                        s0_rec_buffer = s1_rec_buffer
                        s0_rec_buffer_symbolic = s1_rec_buffer_symbolic
                else:
                    if params.dict["recurrent"]:
                        s0_rec_buffer = s1_rec_buffer
            s0 = s1
            a = a1

            if step_counter == eval_length or terminal:
                score_list_raw_reward.append(ep_raw_reward)
                score_list_empirical_constraint_reward.append(
                    ep_empirical_constraint_reward
                )
                score_list_symbolic_constraint_reward.append(
                    ep_symbolic_constraint_reward
                )
                score_list_overall_reward.append(ep_overall_reward)
                break

    mean_raw_reward = np.mean(score_list_raw_reward)
    mean_empirical_constraint_reward = np.mean(score_list_empirical_constraint_reward)
    mean_symbolic_constraint_reward = np.mean(score_list_symbolic_constraint_reward)
    mean_overall_reward = np.mean(score_list_overall_reward)
    print(f"##Actor-{actor_id}")

    training_log_f.write(
        json.dumps(
            {
                "Session": training_session_idx,
                "Epoch": epoch,
                "RawReward": mean_raw_reward,
                "EmpiricalConstraintReward": mean_empirical_constraint_reward,
                "SymbolicConstraintReward": mean_symbolic_constraint_reward,
                "OverallReward": mean_overall_reward,
                "Timestamp": time.time()
            }
        ) + "\n"
    )

    training_log_f.write(f"Epoch: {epoch}, Eval/Return/RawReward: {mean_raw_reward}; ")
    training_log_f.write(
        f"Eval/Return/EmpiricalConstraintReward: {mean_empirical_constraint_reward}; "
    )
    training_log_f.write(
        f"Eval/Return/SymbolicConstraintReward: {mean_symbolic_constraint_reward}; "
    )
    training_log_f.write(f"Eval/Return/OverallReward: {mean_overall_reward}")
    training_log_f.write(f"; Timestamp: {time.time()}\n")
    
    training_log_f.flush()

    return eval_step_counter


class learner_killer:
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
    parser.add_argument(
        "--load", action="store_true", default=False, help="default is  %(default)s"
    )
    parser.add_argument(
        "--eval", action="store_true", default=False, help="default is  %(default)s"
    )
    parser.add_argument("--tb_interval", type=int, default=1)
    parser.add_argument("--train_dir", type=str, default=None)
    parser.add_argument("--mem_r", type=int, default=123456)
    parser.add_argument("--mem_w", type=int, default=12345)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument(
        "--job_name",
        type=str,
        choices=["learner", "actor"],
        required=True,
        help="Job name: either {'learner', actor}",
    )
    parser.add_argument("--task", type=int, required=True, help="Task id")
    parser.add_argument("--model_name", type=str, required=True, help="Experiment id")
    parser.add_argument(
        "--constraints_id", type=int, required=True, help="Constraints id"
    )
    parser.add_argument(
        "--max_actor_epochs",
        type=int,
        required=True,
        help="Set the upper bound of actor epochs",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="threshold hold for the constraint satisfaction -- the delay (large or small) for safety; the noise percentage for robustness",
    )
    parser.add_argument(
        "--x1", type=float, default=5, help="the first changing point for the reward"
    )
    parser.add_argument(
        "--x2", type=float, default=15, help="the second changing point for the reward"
    )
    parser.add_argument(
        "--lambda_",
        type=float,
        default=0.5,
        help="the weight for the reward and the alpha_delta_reward",
    )
    parser.add_argument(
        "--original_model", type=int, default=1, help="Use the original model or not"
    )
    parser.add_argument(
        "--snt_model_wo_ibp",
        type=int,
        default=0,
        help="Use the snt model without IBP or not",
    )
    parser.add_argument(
        "--k_symbolic_components",
        type=int,
        default=1,
        help="The number of symbolic components for training.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="The number of steps to consider for the long horizon constraints.",
    )
    parser.add_argument("--only_tcp", type=int, default=0, help="Use only TCP or not")
    parser.add_argument("--trace_name", type=str, required=True, help="trace name")
    parser.add_argument(
        "--reward_mode",
        type=str,
        required=True,
        help="reward mode, e.g., 'raw-sym', 'heu-sym'",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="random seed for the experiment",
    )
    parser.add_argument(
        "--training_session_idx",
        type=int,
        default=-1,
        help="During training, which training session are we on?",
    )
    ## parameters from parser
    global config
    global params
    config = parser.parse_args()

    print(vars(config))
    
    if config.eval:
        # TODO: check trace_name set when in eval mode
        pass


    lambda_ = config.lambda_
    k_symbolic_components = config.k_symbolic_components
    k = config.k
    only_tcp = False if config.only_tcp == 0 else True
    constraints_id = config.constraints_id

    if only_tcp:
        print(f"---------ONLY_TCP!---------")

    # Set up all the default params
    # SAFETY_CONSTRAINTS_ID: safety on queuing delay; ROBUSTNESS_CONSTRAINTS_ID: robustness on queuing delay.
    assert (
        constraints_id in {
            SAFETY_CONSTRAINTS_ID,
            ROBUSTNESS_CONSTRAINTS_ID,
            LOSS_CONSTRAINTS_ID,
            LOSS_CONSTRAINTS_LIVENESS_ID,
            PERF_ROBUSTNESS_CONSTRAINTS_ID,
            DEEP_BUFFER_CONSTRAINTS,
            SHALLOW_BUFFER_CONSTRAINTS,
            }
    )

    max_actor_epochs = config.max_actor_epochs
    use_original_model = config.original_model == 1
    use_snt_model_wo_ibp = config.snt_model_wo_ibp == 1
    reward_mode = convert_reward_mode(config.reward_mode)
    seed = config.seed

    # TODO: Support more symbolic_modes.
    symbolic_mode = "single_step_only_latency"

    print(f"===================={config.job_name} {config.task}====================")
    print(f"model_name: {config.model_name},")
    print(f"lambda_: {lambda_},")
    print(f"k_symbolic_components: {k_symbolic_components}, ")
    print(f"k: {k},")
    print(f"constraints_id: {constraints_id},")
    print(f"threshold: {config.threshold}")
    print(f"====================Finish Arg Presentation====================")

    # parameters from file
    params = Params(os.path.join(config.base_path, "params.json"))
    print(f"params.dict['num_actors']: {params.dict['num_actors']}")

    if params.dict["single_actor_eval"]:
        local_job_device = ""
        shared_job_device = ""

        def is_actor_fn(i):
            return True

        global_variable_device = "/cpu"
        is_learner = False
        server = tf.train.Server.create_local_server()
        filters = []
    else:
        local_job_device = "/job:%s/task:%d" % (config.job_name, config.task)
        shared_job_device = "/job:learner/task:0"

        is_learner = config.job_name == "learner"
        global_variable_device = shared_job_device + "/cpu"

        def is_actor_fn(i):
            return config.job_name == "actor" and i == config.task

        if params.dict["remote"]:
            cluster = tf.train.ClusterSpec(
                {
                    "actor": params.dict["actor_ip"][: params.dict["num_actors"]],
                    "learner": [params.dict["learner_ip"]],
                }
            )
        else:
            cluster = tf.train.ClusterSpec(
                {
                    "actor": [
                        "localhost:%d" % (8001 + i)
                        for i in range(params.dict["num_actors"])
                    ],
                    "learner": ["localhost:8000"],
                }
            )

        server = tf.train.Server(
            cluster, job_name=config.job_name, task_index=config.task
        )
        filters = [
            shared_job_device,
            local_job_device,
        ]  # Why is this filter not used in Orca's original code?

    print(f"--- Create env----")
    if params.dict["use_TCP"]:
        env_str = "TCP"
        env_peek = TCP_Env_Wrapper(
            env_str, params, use_normalizer=params.dict["use_normalizer"]
        )
    else:
        env_str = "YourEnvironment"
        env_peek = Env_Wrapper(env_str)

    s_dim, a_dim = env_peek.get_dims_info()
    action_scale, action_range = env_peek.get_action_info()

    if not params.dict["use_TCP"]:
        params.dict["state_dim"] = s_dim
    if params.dict["recurrent"]:
        s_dim = s_dim * params.dict["rec_dim"]

    if params.dict["use_hard_target"] == True:
        params.dict["tau"] = 1.0

    with tf.Graph().as_default(), tf.device(local_job_device + "/cpu"):

        tf.set_random_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        actor_op = []
        now = datetime.datetime.now()
        # Create the directory for saving the model.
        model_dir = os.path.join(
            config.base_path,
            params.dict["logdir"],
            f"seed{seed}",
        )
        os.makedirs(model_dir, exist_ok=True)
        tfeventdir = os.path.join(
            model_dir,
            config.job_name + str(config.task) + "-" + config.model_name,
        )
        params.dict["train_dir"] = tfeventdir
        if config.job_name == "actor":
            if config.eval:
                log_directory = f"evaluation_log/seed{seed}"
                experiment_name_suffix = (
                    str(config.model_name)
                    + "-"
                    + config.trace_name
                    + "-"
                    + str(k)
                    + "-"
                    + str(k_symbolic_components)
                )
                writing_mode = "w"
                evaluation_state_log_path = os.path.join(
                    config.base_path,
                    log_directory,
                    f"{experiment_name_suffix+'-EvalStates'}.txt",
                )
                evaluation_reward_and_state_log_path = os.path.join(
                    config.base_path,
                    log_directory,
                    f"{experiment_name_suffix+'-reward_and_state'}.txt",
                )
                os.makedirs(os.path.dirname(evaluation_state_log_path), exist_ok=True)
                os.makedirs(
                    os.path.dirname(evaluation_reward_and_state_log_path), exist_ok=True
                )
                evaluation_state_log_f = open(evaluation_state_log_path, writing_mode)
                evaluation_reward_and_state_log_f = open(
                    evaluation_reward_and_state_log_path, writing_mode
                )
            else:
                log_directory = f"training_log/seed{seed}"
                experiment_name_suffix = (
                    str(config.model_name) + "-" + config.job_name + str(config.task)
                )
                writing_mode = "a"
                evaluation_state_log_f = None
                evaluation_reward_and_state_log_f = None

            evaluation_state_log_path = os.path.join(
                config.base_path,
                log_directory,
                f"{experiment_name_suffix+'-EvalStates'}.txt",
            )
            evaluation_reward_and_state_log_path = os.path.join(
                config.base_path,
                log_directory,
                f"{experiment_name_suffix+'-reward_and_state'}.txt",
            )
            os.makedirs(os.path.dirname(evaluation_state_log_path), exist_ok=True)
            os.makedirs(
                os.path.dirname(evaluation_reward_and_state_log_path), exist_ok=True
            )
            
            if config.eval:
                evaluation_reward_and_state_log_f = open(
                    evaluation_reward_and_state_log_path, writing_mode
                )
            else:
                evaluation_reward_and_state_log_f = None
    
            training_log_path = os.path.join(
                config.base_path, log_directory, f"{experiment_name_suffix}.txt"
            )
            os.makedirs(os.path.dirname(training_log_path), exist_ok=True)
            training_log_f = open(training_log_path, writing_mode)

        if config.eval:
            summary_writer = None
        else:
            if not os.path.exists(tfeventdir):
                os.makedirs(tfeventdir)
            summary_writer = tf.summary.FileWriterCache.get(tfeventdir)

        with tf.device(shared_job_device):
            # Another option: move to v0
            agent = Agent(
                s_dim,
                a_dim,
                batch_size=params.dict["batch_size"],
                summary=summary_writer,
                h1_shape=params.dict["h1_shape"],
                h2_shape=params.dict["h2_shape"],
                stddev=params.dict["stddev"],
                mem_size=params.dict["memsize"],
                gamma=params.dict["gamma"],
                lr_c=params.dict["lr_c"],
                lr_a=params.dict["lr_a"],
                tau=params.dict["tau"],
                PER=params.dict["PER"],
                CDQ=params.dict["CDQ"],
                LOSS_TYPE=params.dict["LOSS_TYPE"],
                noise_type=params.dict["noise_type"],
                noise_exp=params.dict["noise_exp"],
                use_original=use_original_model,
                use_snt_model_wo_ibp=use_snt_model_wo_ibp,
            )
            # agent = Agent(s_dim, a_dim, batch_size=params.dict['batch_size'], summary=summary_writer,h1_shape=params.dict['h1_shape'],
            #             h2_shape=params.dict['h2_shape'],stddev=params.dict['stddev'],mem_size=params.dict['memsize'],gamma=params.dict['gamma'],
            #             lr_c=params.dict['lr_c'],lr_a=params.dict['lr_a'],tau=params.dict['tau'],PER=params.dict['PER'],CDQ=params.dict['CDQ'],
            #             LOSS_TYPE=params.dict['LOSS_TYPE'],noise_type=params.dict['noise_type'],noise_exp=params.dict['noise_exp'])

            dtypes = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
            shapes = [[s_dim], [a_dim], [1], [s_dim], [1]]
            queue = tf.FIFOQueue(10000, dtypes, shapes, shared_name="rp_buf")

        print(f"--- Finished setup----")
        if is_learner:
            with tf.device(params.dict["device"]):
                agent.build_learn()

                agent.create_tf_summary()

            if config.load is True and config.eval == False:
                if os.path.isfile(
                    os.path.join(params.dict["train_dir"], "replay_memory.pkl")
                ):
                    with open(
                        os.path.join(params.dict["train_dir"], "replay_memory.pkl"),
                        "rb",
                    ) as fp:
                        replay_memory = pickle.load(fp)

            _killsignal = learner_killer(agent.rp_buffer)

        for i in range(params.dict["num_actors"]):
            if is_actor_fn(i):
                if params.dict["use_TCP"]:
                    shrmem_r = sysv_ipc.SharedMemory(config.mem_r)
                    shrmem_w = sysv_ipc.SharedMemory(config.mem_w)
                    env = TCP_Env_Wrapper(
                        env_str,
                        params,
                        config=config,
                        for_init_only=False,
                        shrmem_r=shrmem_r,
                        shrmem_w=shrmem_w,
                        use_normalizer=params.dict["use_normalizer"],
                    )
                else:
                    env = GYM_Env_Wrapper(env_str, params)

                a_s0 = tf.placeholder(tf.float32, shape=[s_dim], name="a_s0")
                a_action = tf.placeholder(tf.float32, shape=[a_dim], name="a_action")
                a_reward = tf.placeholder(tf.float32, shape=[1], name="a_reward")
                a_s1 = tf.placeholder(tf.float32, shape=[s_dim], name="a_s1")
                a_terminal = tf.placeholder(tf.float32, shape=[1], name="a_terminal")
                a_buf = [a_s0, a_action, a_reward, a_s1, a_terminal]

                with tf.device(shared_job_device):
                    actor_op.append(queue.enqueue(a_buf))

        if is_learner:
            Dequeue_Length = params.dict["dequeue_length"]
            dequeue = queue.dequeue_many(Dequeue_Length)

        queuesize_op = queue.size()

        if not only_tcp:
            if params.dict["ckptdir"] is not None:
                params.dict["ckptdir"] = os.path.join(
                    config.base_path, params.dict["logdir"], "seed0", params.dict["ckptdir"]
                )
                print("## checkpoint dir:", params.dict["ckptdir"])
                isckpt = os.path.isfile(
                    os.path.join(params.dict["ckptdir"], "checkpoint")
                )
                print("## checkpoint exists?:", isckpt)
                if isckpt == False:
                    print(
                        "\n# # # # # # Warning ! ! ! No checkpoint is loaded, use random model! ! ! # # # # # #\n"
                    )
            else:
                params.dict["ckptdir"] = tfeventdir

            # allow_soft_placement: https://github.com/tensorflow/tensorflow/blob/5bc9d26649cca274750ad3625bd93422617eed4b/tensorflow/core/protobuf/config.proto#L502
            # device_filters: the device not to communicate with.
            tfconfig = tf.ConfigProto(
                allow_soft_placement=True,
                device_filters=filters,
            )

            if params.dict["single_actor_eval"]:
                mon_sess = tf.train.SingularMonitoredSession(
                    checkpoint_dir=params.dict["ckptdir"]
                )
            else:
                mon_sess = tf.train.MonitoredTrainingSession(
                    master=server.target,
                    save_checkpoint_secs=15,
                    save_summaries_secs=None,
                    save_summaries_steps=None,
                    is_chief=is_learner,
                    checkpoint_dir=params.dict["ckptdir"],
                    config=tfconfig,
                    hooks=None,
                )
            agent.assign_sess(mon_sess)

        if is_learner:
            print("=========================Learner is up===================")
            if config.eval:
                while not mon_sess.should_stop():
                    time.sleep(1)
                    continue

            if config.load is False:
                agent.init_target()

            counter = 0
            start = time.time()

            dequeue_thread = threading.Thread(
                target=learner_dequeue_thread,
                args=(agent, params, mon_sess, dequeue, queuesize_op, Dequeue_Length),
                daemon=True,
            )
            first_time = True

            while not mon_sess.should_stop():

                if first_time == True:
                    dequeue_thread.start()
                    first_time = False

                up_del_tmp = params.dict["update_delay"] / 1000.0
                time.sleep(up_del_tmp)
                if agent.rp_buffer.ptr > 200 or agent.rp_buffer.full:
                    agent.train_step()
                    if params.dict["use_hard_target"] == False:
                        agent.target_update()
                        if counter % params.dict["hard_target"] == 0:
                            current_opt_step = agent.sess.run(agent.global_step)
                            logger.info("Optimize step:{}".format(current_opt_step))
                            logger.info("rp_buffer ptr:{}".format(agent.rp_buffer.ptr))
                    else:
                        if counter % params.dict["hard_target"] == 0:
                            agent.target_update()
                            current_opt_step = agent.sess.run(agent.global_step)
                            logger.info("Optimize step:{}".format(current_opt_step))
                            logger.info("rp_buffer ptr:{}".format(agent.rp_buffer.ptr))
                counter += 1

        else:
            print(
                f"=========================Actor{config.task} is up==================="
            )
            start_time = time.time()
            # Initialize the symbolic components.
            if only_tcp is False:
                initial_s1_symbolic = initialize_symbolic_spec(
                    symbolic_mode,
                    threshold=config.threshold,
                    k_symbolic_components=k_symbolic_components,
                    constraints_id=constraints_id,
                )
                if constraints_id == PERF_ROBUSTNESS_CONSTRAINTS_ID:
                    # Perf + robustness
                    union_a_symbolic = {
                        'perf': SymbolicComponent_bound(constraints_id=SAFETY_CONSTRAINTS_ID),
                        'robustness': SymbolicComponent_bound(constraints_id=ROBUSTNESS_CONSTRAINTS_ID),
                    }
                else:
                    union_a_symbolic = SymbolicComponent_bound(
                        constraints_id,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    )
            step_counter = np.int64(0)
            eval_step_counter = np.int64(0)
            info = env.reset()
            s0 = info["state"]
            s0_rec_buffer = np.zeros([s_dim])
            s1_rec_buffer = np.zeros([s_dim])
            s0_rec_buffer[-1 * params.dict["state_dim"] :] = s0
            if only_tcp is False:
                # Initialize the s0_rec_buffer
                s0_rec_buffer_symbolic = update_symbolic_s_single_step(
                    s0_rec_buffer,
                    initial_s1_symbolic,
                    constraints_id,
                )

                if params.dict["recurrent"]:
                    a, a_before_noise = agent.get_concrete_action(
                        s0_rec_buffer, not config.eval
                    )
                    a_symbolic, union_a1_symbolic = get_symbolic_actions(
                        s0_rec_buffer_symbolic, agent, constraints_id
                    )
                else:
                    a, a_before_noise = agent.get_concrete_action(s0, not config.eval)
                a = a[0][0]
            else:
                # only_tcp
                a = 0  # cwnd = 2**alpha * cwnd_tcp, setting alpha to be zero => cwnd = cwnd_tcp

            env.write_action(a)
            past_states = []
            past_actions = []
            past_tcp_cwnds = []
            # for symbolic transitions
            past_cwnds = []

            if constraints_id == PERF_ROBUSTNESS_CONSTRAINTS_ID:
                # Perf + robustness
                union_a_symbolic = {
                    'perf': SymbolicComponent_bound(constraints_id=SAFETY_CONSTRAINTS_ID),
                    'robustness': SymbolicComponent_bound(constraints_id=ROBUSTNESS_CONSTRAINTS_ID),
                }
            else:
                union_a_symbolic = SymbolicComponent_bound(
                    constraints_id,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                )

            epoch = 0
            while epoch < max_actor_epochs:
                # print(f"TRAINING_LOOP {config.job_name} {config.task} - Epoch: {epoch}")
                epoch += 1

                step_counter += 1

                info, r, terminal, error_code = env.step(a, eval_=config.eval)
                s1 = info["state"]
                cwnd_tcp = info["cwnd_tcp"]
                cwnd = get_forced_cwnd_from_alpha_and_tcp_cwnd(a, cwnd_tcp)

                if only_tcp is False:
                    (
                        raw_reward,
                        empirical_constraint_reward,
                        symbolic_constraint_reward,
                        overall_reward,
                        symbolic_cwnd_info,
                        detailed_symbolic_cwnd_info,
                        heuristic_constraints_info,
                    ) = get_raw_and_constraint_reward(
                        state=s0,
                        action=a,
                        action_before_noise=a_before_noise,
                        raw_reward=r,
                        current_tcp_cwnd=cwnd_tcp,
                        constraints_id=constraints_id,
                        reward_mode=reward_mode,
                        threshold=config.threshold,
                        x1=config.x1,
                        x2=config.x2,
                        lambda_=lambda_,
                        union_a_symbolic=union_a_symbolic,
                        past_states=past_states,  # including the current state (effect from the current action)
                        past_actions=past_actions,  # including the current action
                        past_tcp_cwnds=past_tcp_cwnds,  # including the current cwnd_tcp (mapping with the current action)
                        k=k,
                    )
                else:
                    raw_reward = r
                    symbolic_constraint_reward = 0
                    overall_reward = r

                if error_code == True:
                    s1_rec_buffer = np.concatenate(
                        (s0_rec_buffer[params.dict["state_dim"] :], s1)
                    )
                    if not only_tcp:
                        # if (k > 1) and len(past_states) >= 10 + k:
                        #     # TODO: Multi-step symbolic transition
                        #     raise ValueError(
                        #         "Multi-step symbolic transition is not supported."
                        #     )
                        # else:  # single step condition
                        s1_rec_buffer_symbolic = update_symbolic_s_single_step(
                            s1_rec_buffer, initial_s1_symbolic, constraints_id
                        )

                        if params.dict["recurrent"]:
                            a1, a1_before_noise = agent.get_concrete_action(
                                s1_rec_buffer, not config.eval
                            )
                            a1_symbolic, union_a1_symbolic = get_symbolic_actions(
                                s1_rec_buffer_symbolic, agent, constraints_id
                            )
                        else:
                            a1, a1_before_noise = agent.get_concrete_action(
                                s1, not config.eval
                            )
                            raise ImportError(
                                "The recurrent mode is not supported in the evaluation."
                            )

                        a1 = a1[0][0]
                    else:
                        # only_tcp
                        a1 = 0
                    env.write_action(a1)
                else:
                    print("TaskID:" + str(config.task) + "Invalid state received...\n")
                    env.write_action(a)
                    continue

                if config.eval and evaluation_reward_and_state_log_f:
                    evaluation_reward_and_state_log_f.write(
                        f"time: {time.time()}; states: {s0.tolist()}; action: {a}; cwnd_tcp: {cwnd_tcp}; raw_reward: {raw_reward}; certified_reward: {symbolic_constraint_reward}; combined_reward: {overall_reward}; action: {a}\n"
                    )
                    evaluation_reward_and_state_log_f.flush()

                past_actions.append(a)
                past_states.append(s0)
                past_tcp_cwnds.append(cwnd_tcp)
                past_cwnds.append(cwnd)

                if (
                    len(past_states) > 15
                ):  # TODO: Don't bother too long history for now.
                    past_states.pop(0)
                    past_actions.pop(0)
                    past_tcp_cwnds.pop(0)
                    past_cwnds.pop(0)

                # Another option: move to v0
                # assigned_action = a
                if not only_tcp:
                    if agent.use_original or agent.use_snt_model_wo_ibp:
                        assigned_action = a
                    else:
                        assigned_action = np.array([a])
                    if params.dict["recurrent"]:
                        fd = {
                            a_s0: s0_rec_buffer,
                            a_action: assigned_action,
                            a_reward: np.array([overall_reward]),
                            a_s1: s1_rec_buffer,
                            a_terminal: np.array([terminal], np.float),
                        }
                    else:
                        fd = {
                            a_s0: s0,
                            a_action: assigned_action,
                            a_reward: np.array([overall_reward]),
                            a_s1: s1,
                            a_terminal: np.array([terminal], np.float),
                        }

                    if not config.eval:
                        mon_sess.run(actor_op, feed_dict=fd)

                    a_symbolic = a1_symbolic
                    union_a_symbolic = union_a1_symbolic
                    a_before_noise = a1_before_noise
                else:
                    a_before_noise = a1

                s0 = s1
                a = a1

                if only_tcp:
                    if params.dict["recurrent"]:
                        s0_rec_buffer = s1_rec_buffer
                        s0_rec_buffer_symbolic = None
                else:
                    if params.dict["recurrent"]:
                        s0_rec_buffer = s1_rec_buffer
                        s0_rec_buffer_symbolic = s1_rec_buffer_symbolic

                    if not params.dict["use_TCP"] and (terminal):
                        if agent.actor_noise != None:
                            agent.actor_noise.reset()

                    if epoch % params.dict["eval_frequency"] == 0 or (epoch < 1000 and epoch % 50 == 0):
                        eval_step_counter = evaluate_TCP(
                            env,
                            agent,
                            epoch,
                            params,
                            s0,
                            s0_rec_buffer,
                            s0_rec_buffer_symbolic,
                            eval_step_counter,
                            training_log_f,
                            reward_mode,
                            constraints_id,
                            threshold=config.threshold,
                            evaluation_state_log_f=evaluation_state_log_f,
                            x1=config.x1,
                            x2=config.x2,
                            lambda_=lambda_,
                            k_symbolic_components=k_symbolic_components,
                            k=k,
                            evaluation_reward_and_state_log_f=evaluation_reward_and_state_log_f,
                            symbolic_mode=symbolic_mode,
                            actor_id=config.task,
                            eval=config.eval,
                            training_session_idx=config.training_session_idx,
                            start_time=start_time,
                            only_tcp=only_tcp
                        )

                        
        if evaluation_state_log_f:
            evaluation_state_log_f.close()

        if config.eval:
            evaluation_reward_and_state_log_f.close()
        
        training_log_f.close()


def learner_dequeue_thread(
    agent, params, mon_sess, dequeue, queuesize_op, Dequeue_Length
):
    ct = 0
    while True:
        ct = ct + 1
        data = mon_sess.run(dequeue)
        agent.store_many_experience(
            data[0], data[1], data[2], data[3], data[4], Dequeue_Length
        )
        time.sleep(0.01)


def learner_update_thread(agent, params):
    delay = params.dict["update_delay"] / 1000.0
    ct = 0
    while True:
        agent.train_step()
        agent.target_update()
        time.sleep(delay)


if __name__ == "__main__":
    main()
