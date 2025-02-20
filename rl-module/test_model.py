mode=input("What's the training mode? (single/recurrent/recurrent2): ")

from neural_transition import NeuralTransition
import tensorflow as tf

TRAINING_EPOCH_NUM=2001
# TRAINING_EPOCH_NUM=1
DATASET_SIZE=None
# TEST
# TRAINING_EPOCH_NUM=301
# DATASET_SIZE=500

if mode == 'single':
    o_dim=8
elif mode == 'recurrent':
    o_dim=80
elif mode == 'recurrent2':
    o_dim=16
else:
    raise ValueError("Invalid mode: {}".format(mode))

# transition = NeuralTransition(
#     o_dim=o_dim,
#     a_dim=1,
#     h1_shape=256,
#     h2_shape=256,
#     latency_scale=0.5,
#     mode=mode,
# )

# sess = tf.Session()
# tf.global_variables_initializer().run(session=sess)
# transition.assign_sess(sess)
# transition.train_save(epoch_num=TRAINING_EPOCH_NUM, dataset_size=DATASET_SIZE)
# latency_lower_1, latency_upper_1 = transition.get_symbolic_action(
#     o=[[1]*o_dim],
#     o_delta=[[0]*o_dim],
# )
# concrete_latency_1 = transition.get_concrete_action(
#     o=[[1]*o_dim],
# )
# print(f"transition 1's results: {latency_lower_1}, {latency_upper_1}")
# print(f"concrete_latency_1: {concrete_latency_1}")

transition_2 = NeuralTransition(
    o_dim=o_dim,
    a_dim=1,
    h1_shape=256,
    h2_shape=256,
    latency_scale=0.5,
    mode=mode,
)

sess2 = tf.Session()
transition_2.assign_sess(sess2)
transition_2.load_model(epoch_num=TRAINING_EPOCH_NUM-1)
latency_lower_2, latency_upper_2 = transition_2.get_symbolic_action(
    o=[[1]*o_dim],
    o_delta=[[0]*o_dim],
)
concrete_latency_2 = transition_2.get_concrete_action(
    o=[[1]*o_dim],
)
concrete_latency_2_1 = transition_2.get_concrete_action(
    o=[[0]*o_dim],
)
concrete_latency_2_2 = transition_2.get_concrete_action(
    o=[[90]*o_dim],
)
# concrete_latency_1_2 = transition.get_concrete_action(
#     o=[[1]*o_dim],
# )
latency_lower_2, latency_upper_2 = transition_2.get_symbolic_action(
    o=[[1]*o_dim],
    o_delta=[[0]*o_dim],
)
print("----------------Finish Loading and Testing!----------------")
print(f"transition 2's results: {latency_lower_2}, {latency_upper_2}")
print(f"concrete_latency_2: {concrete_latency_2}")
print(f"concrete_latency_2_1: {concrete_latency_2_1}")
print(f"concrete_latency_2_2: {concrete_latency_2_2}")
print(f"latency_lower_2: {latency_lower_2}")
print(f"latency_upper_2: {latency_upper_2}")
# print(f"concrete_latency_1_2: {concrete_latency_1_2}")

# assert(concrete_latency_2 == concrete_latency_1)
# assert(latency_lower_2 == latency_lower_1)

