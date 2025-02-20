import numpy as np
from constants import (
    SymbolicComponent_c_delta,
    SymbolicComponent_bound,
)
from constrained_reward import (
    SAFETY_CONSTRAINTS_ID,
    ROBUSTNESS_CONSTRAINTS_ID,
    LOSS_CONSTRAINTS_ID,
    LOSS_CONSTRAINTS_LIVENESS_ID,
    PERF_ROBUSTNESS_CONSTRAINTS_ID,
    DEEP_BUFFER_CONSTRAINTS,
    SHALLOW_BUFFER_CONSTRAINTS,
)

def initialize_perf_symbolic_spec(
    threshold,
    k,
    constraints_id,
):
    batched_small_s1_delta = []
    batched_small_s1_delta_c = []
    batched_large_s1_delta = []
    batched_large_s1_delta_c = []
    if k>0:
        smaller_threshold = threshold
        # Allow the buffer between small inverseRTT and large inverseRTT.
        larger_threshold = 1 - threshold
        small_width = smaller_threshold / k * 1.0
        large_width = (1 - larger_threshold) / k * 1.0
        spec_state_c = np.array([0.0] * 7)
        spec_state_delta = np.array([0.0] * 7)
        loss_threshold = threshold # if loss rate s[-5] is larger than threshold, we only care these cases.
        # loss: [loss_threshold, 1]
        loss_width = (1 - loss_threshold) / k * 1.0
    for i in range(k):
        # small: [0, smaller_threshold]
        small_s1_delta_c = spec_state_c.copy()
        small_s1_delta = spec_state_delta.copy()
        # Assign the latency component.
        small_s1_delta_c[-2] = (i + 0.5) * small_width
        small_s1_delta[-2] = small_width / 2.0
        if constraints_id == LOSS_CONSTRAINTS_ID or constraints_id == LOSS_CONSTRAINTS_LIVENESS_ID:
            small_s1_delta_c[-5] = (i + 0.5) * loss_width + loss_threshold
            small_s1_delta[-5] = loss_width / 2.0

        batched_small_s1_delta.append(small_s1_delta)
        batched_small_s1_delta_c.append(small_s1_delta_c)

        # large: [larger_threshold, 1]
        large_s1_delta_c = spec_state_c.copy()
        large_s1_delta = spec_state_delta.copy()
        # Assign the latency component.
        large_s1_delta[-2] = large_width / 2.0
        large_s1_delta_c[-2] = (i + 0.5) * large_width + larger_threshold
        if constraints_id == LOSS_CONSTRAINTS_ID or constraints_id == LOSS_CONSTRAINTS_LIVENESS_ID:
            large_s1_delta_c[-5] = (i + 0.5) * loss_width + loss_threshold
            large_s1_delta[-5] = loss_width / 2.0

        batched_large_s1_delta.append(large_s1_delta)
        batched_large_s1_delta_c.append(large_s1_delta_c)
    # s1_delta is an array
    # s1_delta_c is a scalar only representing the latency.

    s1_symbolic = SymbolicComponent_c_delta(
        constraints_id=constraints_id,
        small_c=batched_small_s1_delta_c,
        small_delta=batched_small_s1_delta,
        large_c=batched_large_s1_delta_c,
        large_delta=batched_large_s1_delta,
    )
        
    return s1_symbolic


def initialize_robustness_symbolic_spec(
    threshold,
    k,
    constraints_id,
):
    # Robustness property.
    batched_delta = []
    batched_c = []
    if k>0:
        start_endpoint = -threshold  # [-threshold, threshold]
        width = (
            threshold * 2 / k * 1.0
        )  # The percentage of noise. We now set the noise as [k%, +k%]
        # Later in the main code, we time this abstract noise with the concrete state value.
        spec_state_c = np.array([0.0] * 7)
        spec_state_delta = np.array([0.0] * 7)
    for i in range(k):
        s1_c = spec_state_c.copy()
        s1_delta = spec_state_delta.copy()
        # -2 is for queuing delay.
        s1_c[-2] = (i + 0.5) * width + start_endpoint
        s1_delta[-2] = width / 2.0
        batched_c.append(s1_c)
        batched_delta.append(s1_delta)
    s1_symbolic = SymbolicComponent_c_delta(
        constraints_id=constraints_id,
        c=batched_c,
        delta=batched_delta,
    )
    return s1_symbolic
    

def initialize_perf_symbolic_spec_deep_buffer(
    threshold,
    k,
    constraints_id,
):
    batched_small_s1_delta = []
    batched_small_s1_delta_c = []
    batched_large_s1_delta = []
    batched_large_s1_delta_c = []
    if k>0:
        smaller_threshold = threshold
        # Allow the buffer between small inverseRTT and large inverseRTT.
        larger_threshold = 1 - smaller_threshold
        small_width = smaller_threshold / k * 1.0
        large_width = (1 - larger_threshold) / k * 1.0
        spec_state_c = np.array([0.0] * 7)
        spec_state_delta = np.array([0.0] * 7)
        loss_threshold = threshold # if loss rate s[-5] is larger than threshold, we only care these cases.
        # loss: [loss_threshold, 1]
        loss_width = (1 - loss_threshold) / k * 1.0
    for i in range(k):
        # small: [0, smaller_threshold]
        small_s1_delta_c = spec_state_c.copy()
        small_s1_delta = spec_state_delta.copy()
        # Assign the latency component.
        small_s1_delta_c[-2] = (i + 0.5) * small_width
        small_s1_delta[-2] = small_width / 2.0

        batched_small_s1_delta.append(small_s1_delta)
        batched_small_s1_delta_c.append(small_s1_delta_c)

        # large: [larger_threshold, 1]
        large_s1_delta_c = spec_state_c.copy()
        large_s1_delta = spec_state_delta.copy()
        # Assign the latency component.
        large_s1_delta[-2] = large_width / 2.0
        large_s1_delta_c[-2] = (i + 0.5) * large_width + larger_threshold

        batched_large_s1_delta.append(large_s1_delta)
        batched_large_s1_delta_c.append(large_s1_delta_c)
    # s1_delta is an array
    # s1_delta_c is a scalar only representing the latency.

    s1_symbolic = SymbolicComponent_c_delta(
        constraints_id=constraints_id,
        small_c=batched_small_s1_delta_c,
        small_delta=batched_small_s1_delta,
        large_c=batched_large_s1_delta_c,
        large_delta=batched_large_s1_delta,
    )
        
    return s1_symbolic


def initialize_perf_symbolic_spec_shallow_buffer(
    threshold,
    k,
    constraints_id,
):
    # in shallow buffer, threshold is for loss
    # both small and large inverseRTT are for min queuing delay -> large inverseRTT
    batched_small_s1_delta = []
    batched_small_s1_delta_c = []
    batched_large_s1_delta = []
    batched_large_s1_delta_c = []
    if k>0:
        smaller_threshold = 0.01
        # Allow the buffer between small inverseRTT and large inverseRTT.
        larger_threshold = 1 - smaller_threshold
        small_width = smaller_threshold / k * 1.0
        large_width = (1 - larger_threshold) / k * 1.0
        spec_state_c = np.array([0.0] * 7)
        spec_state_delta = np.array([0.0] * 7)
        loss_threshold = threshold # if loss rate s[-5] is larger than threshold, we only care these cases.
        # loss: [loss_threshold, 1]
        loss_width = (1 - loss_threshold) / k * 1.0
    for i in range(k):
        # small: [0, smaller_threshold]
        small_s1_delta_c = spec_state_c.copy()
        small_s1_delta = spec_state_delta.copy()
        # Assign the latency component.
        small_s1_delta_c[-2] = (i + 0.5) * large_width + larger_threshold
        small_s1_delta[-2] = large_width / 2.0
        small_s1_delta_c[-5] = (i + 0.5) * loss_width + loss_threshold
        small_s1_delta[-5] = loss_width / 2.0

        batched_small_s1_delta.append(small_s1_delta)
        batched_small_s1_delta_c.append(small_s1_delta_c)

        # large: [larger_threshold, 1]
        large_s1_delta_c = spec_state_c.copy()
        large_s1_delta = spec_state_delta.copy()
        # Assign the latency component.
        large_s1_delta[-2] = large_width / 2.0
        large_s1_delta_c[-2] = (i + 0.5) * large_width + larger_threshold

        batched_large_s1_delta.append(large_s1_delta)
        batched_large_s1_delta_c.append(large_s1_delta_c)
    # s1_delta is an array
    # s1_delta_c is a scalar only representing the latency.

    s1_symbolic = SymbolicComponent_c_delta(
        constraints_id=constraints_id,
        small_c=batched_small_s1_delta_c,
        small_delta=batched_small_s1_delta,
        large_c=batched_large_s1_delta_c,
        large_delta=batched_large_s1_delta,
    )
        
    return s1_symbolic


def initialize_symbolic_spec_single_step_only_latency(
    threshold,
    k,
    constraints_id,
):
    """
    Initialize the symbolic parameters for the constraint satisfaction.

    This function is for safety property's precondition.

    params: the parameters for the training.
    threshold: This threshold is only used for loss constraints for shallow buffer. It is used to
    split the queuing delay (inverseRTT) into two parts for deep buffer
    k: the number of the symbolic components.
    """
    if constraints_id == SAFETY_CONSTRAINTS_ID or constraints_id == LOSS_CONSTRAINTS_ID or constraints_id == LOSS_CONSTRAINTS_LIVENESS_ID:
        s1_symbolic = initialize_perf_symbolic_spec(
            threshold=threshold,
            k=k,
            constraints_id=constraints_id,
        )
    elif constraints_id == DEEP_BUFFER_CONSTRAINTS:
        s1_symbolic = initialize_perf_symbolic_spec_deep_buffer(
            threshold=threshold,
            k=k,
            constraints_id=constraints_id,
        )
    elif constraints_id == SHALLOW_BUFFER_CONSTRAINTS:
        s1_symbolic = initialize_perf_symbolic_spec_shallow_buffer(
            threshold=threshold,
            k=k,
            constraints_id=constraints_id,
        )
    elif constraints_id == ROBUSTNESS_CONSTRAINTS_ID:
        s1_symbolic = initialize_robustness_symbolic_spec(
            threshold=threshold,
            k=k,
            constraints_id=ROBUSTNESS_CONSTRAINTS_ID,
        )
    elif constraints_id == PERF_ROBUSTNESS_CONSTRAINTS_ID:
        s1_perf_symbolic = initialize_perf_symbolic_spec(
            threshold=threshold,
            k=k,
            constraints_id=SAFETY_CONSTRAINTS_ID,
        )
        s1_robust_symbolic = initialize_robustness_symbolic_spec(
            threshold=threshold,
            k=k,
            constraints_id=ROBUSTNESS_CONSTRAINTS_ID,
        )
        s1_symbolic = {
            'perf': s1_perf_symbolic,
            'robustness': s1_robust_symbolic,
        }
    else:
        raise ValueError(f"Invalid constraint id {constraints_id}.")

    return s1_symbolic


def single_step_symbolic_transition_only_latency(
    s1_rec_buffer,
    params,
    batched_small_s1_delta,
    batched_small_s1_delta_c,
    batched_large_s1_delta,
    batched_large_s1_delta_c,
):
    # Attention: array assignment.
    all_small_s1_delta_rec_buffer = []
    all_small_s1_c_rec_buffer = []
    all_large_s1_delta_rec_buffer = []
    all_large_s1_c_rec_buffer = []
    for small_s1_delta, small_s1_delta_c, large_s1_delta, large_s1_delta_c in zip(
        batched_small_s1_delta,
        batched_small_s1_delta_c,
        batched_large_s1_delta,
        batched_large_s1_delta_c,
    ):
        small_s1_delta_rec_buffer = np.concatenate(
            (np.zeros_like(s1_rec_buffer[params.dict["state_dim"] :]), small_s1_delta)
        )
        small_s1_c_rec_buffer = s1_rec_buffer.copy()
        small_s1_c_rec_buffer[-2] = small_s1_delta_c
        large_s1_delta_rec_buffer = np.concatenate(
            (np.zeros_like(s1_rec_buffer[params.dict["state_dim"] :]), large_s1_delta)
        )
        large_s1_c_rec_buffer = s1_rec_buffer.copy()
        large_s1_c_rec_buffer[-2] = large_s1_delta_c
        all_small_s1_delta_rec_buffer.append(small_s1_delta_rec_buffer)
        all_small_s1_c_rec_buffer.append(small_s1_c_rec_buffer)
        all_large_s1_delta_rec_buffer.append(large_s1_delta_rec_buffer)
        all_large_s1_c_rec_buffer.append(large_s1_c_rec_buffer)
    # TODO: Do not use batched version for now. The symbolic results from batches are wrong.
    return (
        all_small_s1_delta_rec_buffer,
        all_small_s1_c_rec_buffer,
        all_large_s1_delta_rec_buffer,
        all_large_s1_c_rec_buffer,
    )


def multi_step_symbolic_transition_only_latency(
    s1_rec_buffer,
    params,
    batched_small_s1_delta,  # The delta of one step spec.
    batched_small_s1_delta_c,
    batched_large_s1_delta,
    batched_large_s1_delta_c,
    m_steps,  # For m_steps' last latency, we replace it with the symbolic representation.
):
    """
    Given a concrete past states, we replace the last m steps of latency with the symbolic representation.
    For each symbolic representation, the symbolic representation maintain the same.
    """
    # Attention: array assignment.
    all_small_s1_delta_rec_buffer = []
    all_small_s1_c_rec_buffer = []
    all_large_s1_delta_rec_buffer = []
    all_large_s1_c_rec_buffer = []
    single_step_state_size = params.dict["state_dim"]
    symbolic_indices = [-2 - i * single_step_state_size for i in range(m_steps)]
    replaced_variable_index = single_step_state_size * m_steps
    assert replaced_variable_index <= len(s1_rec_buffer)

    for small_s1_delta, small_s1_delta_c, large_s1_delta, large_s1_delta_c in zip(
        batched_small_s1_delta,
        batched_small_s1_delta_c,
        batched_large_s1_delta,
        batched_large_s1_delta_c,
    ):
        small_s1_delta_rec_buffer = np.concatenate(
            (
                np.zeros_like(s1_rec_buffer[replaced_variable_index:]),
                np.tile(small_s1_delta, m_steps),
            )
        )
        small_s1_c_rec_buffer = s1_rec_buffer.copy()
        # Assign the center of the symbolic states for all the selected indices: the past ones representing latency.
        small_s1_c_rec_buffer.put(symbolic_indices, small_s1_delta_c)
        large_s1_delta_rec_buffer = np.concatenate(
            (
                np.zeros_like(s1_rec_buffer[replaced_variable_index:]),
                np.tile(large_s1_delta, m_steps),
            )
        )
        large_s1_c_rec_buffer = s1_rec_buffer.copy()
        # Assign the center of the symbolic states for all the selected indices: the past ones representing latency.
        large_s1_c_rec_buffer.put(symbolic_indices, large_s1_delta_c)

        all_small_s1_c_rec_buffer.append(small_s1_c_rec_buffer)
        all_small_s1_delta_rec_buffer.append(small_s1_delta_rec_buffer)
        all_large_s1_c_rec_buffer.append(large_s1_c_rec_buffer)
        all_large_s1_delta_rec_buffer.append(large_s1_delta_rec_buffer)
    # TODO: Not using the batched version for now. The symbolic results from batches are wrong.
    return (
        all_small_s1_delta_rec_buffer,
        all_small_s1_c_rec_buffer,
        all_large_s1_delta_rec_buffer,
        all_large_s1_c_rec_buffer,
    )

def update_perf_symbolic_s_single_step(
    s1_rec_buffer,
    s1_symbolic,
    constraints_id,
):
    s1_rec_buffer_symbolic = SymbolicComponent_c_delta(
        constraints_id=constraints_id,
        small_c=[],
        small_delta=[],
        large_c=[],
        large_delta=[],
    )
    for small_s1_delta, small_s1_delta_c, large_s1_delta, large_s1_delta_c in zip(
        s1_symbolic.small_delta,
        s1_symbolic.small_c,
        s1_symbolic.large_delta,
        s1_symbolic.large_c,
    ):
        small_s1_delta_rec_buffer = np.concatenate(
            (np.zeros_like(s1_rec_buffer[7:]), small_s1_delta)
        )
        small_s1_c_rec_buffer = s1_rec_buffer.copy()
        small_s1_c_rec_buffer[-2] = small_s1_delta_c[-2]
        large_s1_delta_rec_buffer = np.concatenate(
            (np.zeros_like(s1_rec_buffer[7:]), large_s1_delta)
        )
        large_s1_c_rec_buffer = s1_rec_buffer.copy()
        large_s1_c_rec_buffer[-2] = large_s1_delta_c[-2]
        s1_rec_buffer_symbolic.small_delta.append(small_s1_delta_rec_buffer)
        s1_rec_buffer_symbolic.small_c.append(small_s1_c_rec_buffer)
        s1_rec_buffer_symbolic.large_delta.append(large_s1_delta_rec_buffer)
        s1_rec_buffer_symbolic.large_c.append(large_s1_c_rec_buffer)

    return s1_rec_buffer_symbolic


def update_robustness_symbolic_s_single_step(
        s1_rec_buffer,
        s1_symbolic,
        constraints_id,
    ):
    s1_rec_buffer_symbolic = SymbolicComponent_c_delta(
        constraints_id=constraints_id,
        c=[],
        delta=[],
    )
    current_queuing_delay_value = s1_rec_buffer[-2]
    for delta, c in zip(s1_symbolic.delta, s1_symbolic.c):
        # We use percentage for noises.
        # noise: [0, 0.05]=[0, 5%] each component (delta, c) = [0, n1], ..., [nk, 5%]
        # With percentage noise, final_noise = c*symbolic_noise
        # each component = [0, c*n1], ... [c*nk, c*5%]
        tmp_delta = delta[-2] * current_queuing_delay_value
        tmp_c = current_queuing_delay_value + c[-2] * current_queuing_delay_value
        tmp_l = tmp_c - tmp_delta
        tmp_u = tmp_c + tmp_delta
        if tmp_l < 0:
            tmp_l = 0.0
        if tmp_u > 1:
            tmp_u = 1.0
        tmp_delta = (tmp_u - tmp_l) / 2.0
        tmp_c = (tmp_l + tmp_u) / 2.0
        delta_rec_buffer = np.concatenate((np.zeros_like(s1_rec_buffer[7:]), delta))
        delta_rec_buffer[-2] = tmp_delta
        c_rec_buffer = s1_rec_buffer.copy()
        c_rec_buffer[-2] = tmp_c
        s1_rec_buffer_symbolic.delta.append(delta_rec_buffer)
        s1_rec_buffer_symbolic.c.append(c_rec_buffer)
    
    return s1_rec_buffer_symbolic

def update_symbolic_s_single_step(
    s1_rec_buffer,
    s1_symbolic,
    constraints_id,
):
    # Attention: array assignment.
    if constraints_id in {
        SAFETY_CONSTRAINTS_ID,
        LOSS_CONSTRAINTS_ID,
        LOSS_CONSTRAINTS_LIVENESS_ID,
        DEEP_BUFFER_CONSTRAINTS,
        SHALLOW_BUFFER_CONSTRAINTS,
    }:
        s1_rec_buffer_symbolic = update_perf_symbolic_s_single_step(
            s1_rec_buffer,
            s1_symbolic,
            constraints_id,
        )
    elif constraints_id == ROBUSTNESS_CONSTRAINTS_ID:
        s1_rec_buffer_symbolic = update_robustness_symbolic_s_single_step(
            s1_rec_buffer,
            s1_symbolic,
            constraints_id,
        )
    elif constraints_id == PERF_ROBUSTNESS_CONSTRAINTS_ID:
        s1_perf_rec_buffer_symbolic = update_perf_symbolic_s_single_step(
            s1_rec_buffer,
            s1_symbolic['perf'],
            constraints_id=SAFETY_CONSTRAINTS_ID,
        )
        s1_robust_rec_buffer_symbolic = update_robustness_symbolic_s_single_step(
            s1_rec_buffer,
            s1_symbolic['robustness'],
            constraints_id=ROBUSTNESS_CONSTRAINTS_ID,
        )
        s1_rec_buffer_symbolic = {
            'perf': s1_perf_rec_buffer_symbolic,
            'robustness': s1_robust_rec_buffer_symbolic,
        }
    else:
        raise ValueError(f"Invalid constraint id {constraints_id}.")

    # TODO: Do not use batched version for now. The batched symbolic results from IBP are wrong.
    return s1_rec_buffer_symbolic


def get_perf_symbolic_actions(
    s1_rec_buffer_symbolic,
    agent,
    constraints_id,
):
    a1_symbolic = SymbolicComponent_bound(
        constraints_id=constraints_id,
        small_l=[],
        small_u=[],
        large_l=[],
        large_u=[],
    )
    for (
        one_component_small_s1_c_rec_buffer,
        one_component_small_s1_delta_rec_buffer,
        one_component_large_s1_c_rec_buffer,
        one_component_large_s1_delta_rec_buffer,
    ) in zip(
        s1_rec_buffer_symbolic.small_c,
        s1_rec_buffer_symbolic.small_delta,
        s1_rec_buffer_symbolic.large_c,
        s1_rec_buffer_symbolic.large_delta,
    ):
        (
            one_component_small_inverseRTT_a1_lower,
            one_component_small_inverseRTT_a1_upper,
        ) = agent.get_symbolic_action(
            s=one_component_small_s1_c_rec_buffer,
            s_delta=one_component_small_s1_delta_rec_buffer,
            use_noise=False,
        )
        (
            one_component_large_inverseRTT_a1_lower,
            one_component_large_inverseRTT_a1_upper,
        ) = agent.get_symbolic_action(
            s=one_component_large_s1_c_rec_buffer,
            s_delta=one_component_large_s1_delta_rec_buffer,
            use_noise=False,
        )
        a1_symbolic.small_l.append(one_component_small_inverseRTT_a1_lower[0][0])
        a1_symbolic.small_u.append(one_component_small_inverseRTT_a1_upper[0][0])
        a1_symbolic.large_l.append(one_component_large_inverseRTT_a1_lower[0][0])
        a1_symbolic.large_u.append(one_component_large_inverseRTT_a1_upper[0][0])
    union_a1_symbolic = SymbolicComponent_bound(
        constraints_id=constraints_id,
        small_l=min(a1_symbolic.small_l) if a1_symbolic.small_l else 0.0,
        small_u=max(a1_symbolic.small_u) if a1_symbolic.small_u else 0.0,
        large_l=min(a1_symbolic.large_l) if a1_symbolic.large_l else 0.0,
        large_u=max(a1_symbolic.large_u) if a1_symbolic.large_u else 0.0,
    )
    
    return a1_symbolic, union_a1_symbolic


def get_robustness_symbolic_actions(
    s1_rec_buffer_symbolic,
    agent,
    constraints_id,
):
    a1_symbolic = SymbolicComponent_bound(
        constraints_id=constraints_id,
        l=[],
        u=[],
    )
    for (one_component_c_rec_buffer, one_component_delta_rec_buffer,) in zip(
        s1_rec_buffer_symbolic.c,
        s1_rec_buffer_symbolic.delta,
    ):
        (
            one_component_a1_lower,
            one_component_a1_upper,
        ) = agent.get_symbolic_action(
            s=one_component_c_rec_buffer,
            s_delta=one_component_delta_rec_buffer,
            use_noise=False,
        )
        a1_symbolic.l.append(one_component_a1_lower[0][0])
        a1_symbolic.u.append(one_component_a1_upper[0][0])
    union_a1_symbolic = SymbolicComponent_bound(
        constraints_id=constraints_id,
        l=min(a1_symbolic.l) if a1_symbolic.l else 0.0,
        u=max(a1_symbolic.u) if a1_symbolic.u else 0.0,
    )
    
    return a1_symbolic, union_a1_symbolic
    

def get_symbolic_actions(
    s1_rec_buffer_symbolic,
    agent,
    constraints_id,
):
    """
    s1_rec_buffer_symbolic: list of symbolic s1_rec_buffer components
    """
    # TODO: The batched version in IBP seems troublesome. I now unfold the batched
    # c and delta and do all the computation in a sequential way.
    # TODO: To keep all the detailed boundary of a1 here and pass down to the network model.
    # a1 is monotonic to the cwnd.
    if constraints_id in {
        SAFETY_CONSTRAINTS_ID,
        LOSS_CONSTRAINTS_ID,
        LOSS_CONSTRAINTS_LIVENESS_ID,
        DEEP_BUFFER_CONSTRAINTS,
        SHALLOW_BUFFER_CONSTRAINTS,
        }:
        a1_symbolic, union_a1_symbolic = get_perf_symbolic_actions(
            s1_rec_buffer_symbolic,
            agent,
            constraints_id,
        )
    elif constraints_id == ROBUSTNESS_CONSTRAINTS_ID:
        a1_symbolic, union_a1_symbolic = get_robustness_symbolic_actions(
            s1_rec_buffer_symbolic,
            agent,
            constraints_id,
        )
    elif constraints_id == PERF_ROBUSTNESS_CONSTRAINTS_ID:
        a1_perf_symbolic, union_a1_perf_symbolic = get_perf_symbolic_actions(
            s1_rec_buffer_symbolic['perf'],
            agent,
            constraints_id=SAFETY_CONSTRAINTS_ID,
        )
        a1_robust_symbolic, union_a1_robust_symbolic = get_robustness_symbolic_actions(
            s1_rec_buffer_symbolic['robustness'],
            agent,
            constraints_id=ROBUSTNESS_CONSTRAINTS_ID,
        )
        a1_symbolic = {
            'perf': a1_perf_symbolic,
            'robustness': a1_robust_symbolic,
        }
        union_a1_symbolic = {
            'perf': union_a1_perf_symbolic,
            'robustness': union_a1_robust_symbolic,
        }
    else:
        raise ValueError(f"constraints_id {constraints_id} is not supported.")
    return a1_symbolic, union_a1_symbolic


def is_empty(s_rec_buffer_symbolic):
    small_delta = s_rec_buffer_symbolic.small_delta
    large_delta = s_rec_buffer_symbolic.large_delta
    # if the largest value in small_delta <= 0.0:
    # print(f"small_delta: {small_delta}")
    # print(f"large_delta: {large_delta}")
    if len(small_delta) <= 0 or len(large_delta) <= 0:
        return True
    else:
        return False
