from collections import deque
from utils_v2 import get_forced_cwnd_from_alpha_and_tcp_cwnd, get_symbolic_forced_cwnd

NO_CONSTRAINED = "NO_CONSTRAINED"
ONLY_EMPIRICAL_CONSTRAINED = "ONLY_EMPIRICAL_CONSTRAINED"
ONLY_SYMBOLIC_CONSTRAINED = "ONLY_SYMBOLIC_CONSTRAINED"
EMPIRICAL_CONSTRAINED = "EMPIRICAL_CONSTRAINED"
SYMBOLIC_CONSTRAINED = "SYMBOLIC_CONSTRAINED"
EMPIRICAL_AND_SYMBOLIC_CONSTRAINED = "EMPIRICAL_AND_SYMBOLIC_CONSTRAINED"
RAW_AND_SYMBOLIC_CONSTRAINED = "RAW_AND_SYMBOLIC_CONSTRAINED"

SAFETY_CONSTRAINTS_ID = 6
ROBUSTNESS_CONSTRAINTS_ID = 7
LOSS_CONSTRAINTS_ID = 8
LOSS_CONSTRAINTS_LIVENESS_ID = 9
PERF_ROBUSTNESS_CONSTRAINTS_ID = 10 # performance and robustness constraints together
DEEP_BUFFER_CONSTRAINTS = 11
SHALLOW_BUFFER_CONSTRAINTS = 12

reward_q = deque(maxlen=3) # look back 3 steps

low_delay_delta_cwnd_lower_q = deque(maxlen=3)
low_delay_delta_cwnd_upper_q = deque(maxlen=3)
high_delay_delta_cwnd_lower_q = deque(maxlen=3)
high_delay_delta_cwnd_upper_q = deque(maxlen=3)
# placeholder
low_delay_delta_cwnd_lower_q.append(0)
low_delay_delta_cwnd_upper_q.append(0)
high_delay_delta_cwnd_lower_q.append(0)
high_delay_delta_cwnd_upper_q.append(0)



def convert_reward_mode(reward_mode_name):
    if reward_mode_name == "heu-sym":
        return EMPIRICAL_AND_SYMBOLIC_CONSTRAINED
    elif reward_mode_name == "raw-sym":
        return RAW_AND_SYMBOLIC_CONSTRAINED
    elif reward_mode_name == "heu-only":
        return ONLY_EMPIRICAL_CONSTRAINED
    elif reward_mode_name == "heu":
        return EMPIRICAL_CONSTRAINED
    elif reward_mode_name == "sym-only":
        return ONLY_SYMBOLIC_CONSTRAINED
    elif reward_mode_name == "sym":
        return SYMBOLIC_CONSTRAINED
    elif reward_mode_name == "raw":
        return NO_CONSTRAINED

    raise ValueError(
        f"reward_mode_name={reward_mode_name} is not associated with any specific reward type. Please double check."
    )


def constraints_two_side_liveness_regulate_final_cwnd(
    state,
    action,
    current_tcp_cwnd,
    previous_action,
    previous_tcp_cwnd,
    threshold=0.5,
    x1=5,
    x2=25,
):
    """
    Liveness constraint:
    if latency < k1, we want cwnd to increase.
    if latency > k2, we want cwnd to decrease.
    => cwnd = 2**alpha * cwnd_TCP [in paper]
    => in implementation: cwnd = int(math.pow(4, alpha) * 100) * cwnd_TCP / 100
    +
    the abs(delta_cwnd) should be within a range.
    the abs(delta_action) -- abs(delta_alpha) should be within a range.
    """
    inverseRTT = state[-2]  # inverseRTT = min_rtt / srtt
    current_forced_cwnd = get_forced_cwnd_from_alpha_and_tcp_cwnd(
        action, current_tcp_cwnd
    )
    previous_forced_cwnd = get_forced_cwnd_from_alpha_and_tcp_cwnd(
        previous_action, previous_tcp_cwnd
    )
    delta_cwnd = current_forced_cwnd - previous_forced_cwnd
    reward = 0
    if inverseRTT > threshold:  # RTT is small, the cwnd should increase.
        if delta_cwnd <= x1:
            reward = 1 / x1 * delta_cwnd
        else:  # delta_cwnd > x1
            reward = 1 / (x1 - x2) * delta_cwnd - x2 / (x1 - x2)
        # Not use for now.
        if delta_cwnd < 0 or delta_cwnd > x2:
            reward = 0
    else:  # inverseRTT < threshold, RTT is large, the cwnd should decrease.
        if delta_cwnd >= -x1:
            reward = -1 / x1 * delta_cwnd
        else:  # delta_cwnd < -x1
            reward = -1 / (x1 - x2) * delta_cwnd - x2 / (x1 - x2)
        if delta_cwnd > 0 or delta_cwnd < -x2:
            reward = 0

    latency_signal = "mid"
    if inverseRTT > threshold:
        latency_signal = "small"
    elif inverseRTT < 1 - threshold:
        latency_signal = "large"
    heuristic_constraints_info = {
        "latency_signal": latency_signal,
        "delta_cwnd": delta_cwnd,
    }
    return reward, heuristic_constraints_info


def long_horizon_constraints_latency(
    s,
    a,
    tcp_cwnd,
    past_states=None,  # list of states
    past_actions=None,  # list of actions
    past_tcp_cwnds=None,  # list of tcp_cwnds
    k1=1,
    k2=1,
    threshold=0.5,
    x1=0.05,
    x2=0.2,
):
    """
    Longer horizon constraints for latency.
    If latency is large (inverseRTT is small <= threshold) for k1 steps, the cwnd should decrease within k2 steps.
    delta cwnd is [-x2, 0], peak at [-x1]
    If latency is small (inverseRTT is larger >= 1 - threshold) for k1 steps, the cwnd should increase within k2 steps.
    delta cwnd is [0, x2], peak at [x1]
    """
    # Quicker convergence, the better.
    # if past k1 steps, inverseRTT <= threshold
    # in the past k2 steps and the current step, the cwnd should be decreasing
    # the decreasing does not need to be shrinking?
    # The start of the decreasing, the quicker, the better.
    # For the inverseRTT >= threshold case, the same.
    assert k2 <= k1

    current_inverseRTT = s[-2]
    current_forced_cwnd = get_forced_cwnd_from_alpha_and_tcp_cwnd(a, tcp_cwnd)
    previous_inverseRTTs = [s[-2] for s in past_states]
    previous_forced_cwnds = [
        get_forced_cwnd_from_alpha_and_tcp_cwnd(a, c)
        for a, c in zip(past_actions, past_tcp_cwnds)
    ]
    previous_delta_cwnds = [
        p - c for c, p in zip(previous_forced_cwnds[:-1], previous_forced_cwnds[1:])
    ]
    current_delta_cwnd = current_forced_cwnd - previous_forced_cwnds[-1]

    reward = 0

    def extract_reward_from_decreasing_delta_cwnd_x1_x2(delta_cwnd, x1, x2):
        if delta_cwnd >= -x1:
            current_reward = -1 / x1 * delta_cwnd
        else:
            current_reward = -1 / (x1 - x2) * delta_cwnd - x2 / (x1 - x2)
        if delta_cwnd > 0 or delta_cwnd < -x2:
            current_reward = 0
        return current_reward

    def extract_reward_from_increasing_delta_cwnd_x1_x2(delta_cwnd, x1, x2):
        if delta_cwnd <= x1:
            current_reward = 1 / x1 * delta_cwnd
        else:
            current_reward = 1 / (x1 - x2) * delta_cwnd - x2 / (x1 - x2)
        if delta_cwnd < 0 or delta_cwnd > x2:
            current_reward = 0
        return current_reward

    latency_signal = "mid"

    # if for all the past k1 consecutive steps: inverseRTT <= threshold <=> the latency is large
    if (
        sum([1 for rtt in previous_inverseRTTs[-k1:] if rtt <= threshold]) >= k1
        and current_inverseRTT <= threshold
    ):
        # The past k1 steps' latency is large and the current step's latency is large.
        # The cwnd should decrease for the last k2 steps.
        # For the last k2 steps, if all decreasing great
        # if just some of them decreasing, we want the ones close to the end decreasing.
        reward = extract_reward_from_decreasing_delta_cwnd_x1_x2(
            current_delta_cwnd, x1, x2
        )
        for i, delta_cwnd in enumerate(previous_delta_cwnds[-k2:]):
            # Closer to the current step, the more important.
            reward += (
                extract_reward_from_decreasing_delta_cwnd_x1_x2(delta_cwnd, x1, x2)
                * i
                / k2
            )
        latency_signal = "large"
    elif (
        sum([1 for rtt in previous_inverseRTTs[-k1:] if rtt >= 1 - threshold]) >= k1
        and current_inverseRTT >= 1 - threshold
    ):
        # just one time large or small latency, let's don't care about it.
        # The past k1 steps' latency is small and the current step's latency is small.
        # The cwnd should increase for the last k2 steps.
        # For the last k2 steps, if all increasing great
        # if just some of them increasing, we want the ones close to the end increasing.
        reward = extract_reward_from_increasing_delta_cwnd_x1_x2(
            current_delta_cwnd, x1, x2
        )
        for i, delta_cwnd in enumerate(previous_delta_cwnds[-k2:]):
            # Closer to the current step, the more important.
            reward += (
                extract_reward_from_increasing_delta_cwnd_x1_x2(delta_cwnd, x1, x2)
                * i
                / k2
            )
        latency_signal = "small"
    else:
        # Other case, get back to the single step spec
        if current_inverseRTT > threshold:
            reward = extract_reward_from_increasing_delta_cwnd_x1_x2(
                current_delta_cwnd, x1, x2
            )
        elif current_inverseRTT < 1 - threshold:
            reward = extract_reward_from_decreasing_delta_cwnd_x1_x2(
                current_delta_cwnd, x1, x2
            )

    heuristic_constraints_info = {
        "latency_signal": latency_signal,
        "delta_cwnd": current_delta_cwnd,
    }
    return reward, heuristic_constraints_info


def get_empirical_constrained_reward(
    state,
    action,
    current_tcp_cwnd,
    previous_action,
    previous_tcp_cwnd,
    constraints_id=0,
    threshold=0.5,
    x1=5,
    x2=25,
    lambda_=0.5,
    past_states=None,
    past_actions=None,
    past_tcp_cwnds=None,
    k=2,
):
    """
    Extract the constrained reward from the state, action, current_tcp_cwnd,
    previous_action, previous_tcp_cwnd, and constraints_id.

    Args:
        state: state
        action: action
        current_tcp_cwnd: current tcp cwnd
        previous_action: previous action
        previous_tcp_cwnd: previous tcp cwnd
        constraints_id: id of the constraint to use
        threshold: threshold value for the constraint
        x1: the first changing point for the reward
        x2: the second changing point for the reward
        lambda_: the weight for the reward and the alpha_delta_reward
        k: the number of steps to consider for the long horizon constraints

    Returns:
        constrained reward
    """
    if constraints_id == 6:
        # The single step spec heuristic reward.
        return constraints_two_side_liveness_regulate_final_cwnd(
            state=state,
            action=action,
            current_tcp_cwnd=current_tcp_cwnd,
            previous_action=previous_action,
            previous_tcp_cwnd=previous_tcp_cwnd,
            threshold=threshold,
            x1=x1,
            x2=x2,
        )
    elif constraints_id == 16:
        # The multi step spec heuristic reward.
        return long_horizon_constraints_latency(
            s=state,
            a=action,
            tcp_cwnd=current_tcp_cwnd,
            past_states=past_states,
            past_actions=past_actions,
            past_tcp_cwnds=past_tcp_cwnds,
            threshold=threshold,
            x1=x1,
            x2=x2,
            k1=k,
            k2=1,
        )
    else:
        return 0.0, {
            "latency_signal": 0.0,
            "delta_cwnd": 0.0,
        }


def calculate_distance(l, r, target_l, target_u):
    if l > target_u or r < target_l:
        # if l > target_u:
        #     return -(l-target_u) / (r-target_u)
        # else:
        #     return -(target_l-r) / (target_l-l)
        return 0
    elif l >= target_l and r <= target_u:
        return 1
    else:
        overlapped_area = min(r, target_u) - max(l, target_l)
    return overlapped_area / (r - l)


def calculate_safety_reward(cwnd_l, cwnd_r, previous_cwnd, target_l, target_u):
    # print(f"cwnd_l, cwnd_r: {cwnd_l, cwnd_r}")
    # Attention! delta cwnd = current cwnd - previous cwnd
    delta_cwnd_l, delta_cwnd_r = cwnd_l - previous_cwnd, cwnd_r - previous_cwnd
    # We want to make [delta_cwnd_l, delta_cwnd_r] close to [target_l, target_u]
    # if delta_cwnd# in target#, the reward is 1
    # if delta_cwnd# is far from target#, the reward is close to 0
    delta_cwnd_area = delta_cwnd_r - delta_cwnd_l
    if delta_cwnd_l > target_u or delta_cwnd_r < target_l:
        # if delta_cwnd is completely out of target
        return 0
    elif delta_cwnd_l >= target_l and delta_cwnd_r <= target_u:
        # if delta_cwnd is completely in target
        return 1
    else:
        # if delta_cwnd is partly in target
        overlapped_area = min(delta_cwnd_r, target_u) - max(delta_cwnd_l, target_l)
    return overlapped_area / delta_cwnd_area


def get_cwnd_bound(
    small_inverseRTT_a1_lower,
    small_inverseRTT_a1_upper,
    large_inverseRTT_a1_lower,
    large_inverseRTT_a1_upper,
    current_tcp_cwnd,
):
    (
        small_inverseRTT_forced_cwnd_lower,
        small_inverseRTT_forced_cwnd_upper,
    ) = get_symbolic_forced_cwnd(
        small_inverseRTT_a1_lower, small_inverseRTT_a1_upper, current_tcp_cwnd
    )
    # When inverseRTT is small -> latency is large -> decrease cwnd -> delta cwnd negative
    # -> symbolic_cwnd - precious_cwnd < 0
    # want to make the symbolic_cwnd - previous_cwnd in [-x2, 0]
    (
        large_inverseRTT_forced_cwnd_lower,
        large_inverseRTT_forced_cwnd_upper,
    ) = get_symbolic_forced_cwnd(
        large_inverseRTT_a1_lower, large_inverseRTT_a1_upper, current_tcp_cwnd
    )  # want to make the symbolic_cwnd - previous_cwnd in [0, x2]

    return (
        small_inverseRTT_forced_cwnd_lower,
        small_inverseRTT_forced_cwnd_upper,
        large_inverseRTT_forced_cwnd_lower,
        large_inverseRTT_forced_cwnd_upper,
    )

def get_perf_symbolic_constrained_reward(
    state,
    action,
    action_before_noise,
    current_tcp_cwnd,
    previous_action,
    previous_tcp_cwnd,
    threshold,
    x1,
    x2,
    lambda_,
    union_a_symbolic,
    constraints_id,
    a_symbolic=None,
    k=1,
):
    # Get the symbolic version of concrete constraints of constraints_id == 6.
    # if inverseRTT > threshold: delta_cwnd should be in [0, x2]
    # if inverseRTT < threshold: delta_cwnd should be in [-x2, 0]
    # if constraints_id == 8,
    # the delta_cwnd should be in [-x2, 0]. No matter the latency, as long as the loss rate is large,
    # we want the cwnd to decrease.
    small_inverseRTT_a1_lower = union_a_symbolic.small_l
    small_inverseRTT_a1_upper = union_a_symbolic.small_u
    large_inverseRTT_a1_lower = union_a_symbolic.large_l
    large_inverseRTT_a1_upper = union_a_symbolic.large_u

    previous_forced_cwnd = get_forced_cwnd_from_alpha_and_tcp_cwnd(
        previous_action, previous_tcp_cwnd
    )
    (
        small_inverseRTT_forced_cwnd_lower,
        small_inverseRTT_forced_cwnd_upper,
        large_inverseRTT_forced_cwnd_lower,
        large_inverseRTT_forced_cwnd_upper,
    ) = get_cwnd_bound(
        small_inverseRTT_a1_lower,
        small_inverseRTT_a1_upper,
        large_inverseRTT_a1_lower,
        large_inverseRTT_a1_upper,
        current_tcp_cwnd,
    )
    if a_symbolic:
        detailed_symbolic_cwnd_info = {
            "small_inverseRTT_delta_cwnd_lower": [],
            "small_inverseRTT_delta_cwnd_upper": [],
            "large_inverseRTT_delta_cwnd_lower": [],
            "large_inverseRTT_delta_cwnd_upper": [],
        }
        for (
            one_component_small_l,
            one_component_small_u,
            one_component_large_l,
            one_component_large_u,
        ) in zip(
            a_symbolic.small_l,
            a_symbolic.small_u,
            a_symbolic.large_l,
            a_symbolic.large_u,
        ):
            (
                one_component_small_cwnd_l,
                one_component_small_cwnd_u,
                one_component_large_cwnd_l,
                one_component_large_cwnd_u,
            ) = get_cwnd_bound(
                one_component_small_l,
                one_component_small_u,
                one_component_large_l,
                one_component_large_u,
                current_tcp_cwnd,
            )
            detailed_symbolic_cwnd_info["small_inverseRTT_delta_cwnd_lower"].append(
                one_component_small_cwnd_l - previous_forced_cwnd
            )
            detailed_symbolic_cwnd_info["small_inverseRTT_delta_cwnd_upper"].append(
                one_component_small_cwnd_u - previous_forced_cwnd
            )
            detailed_symbolic_cwnd_info["large_inverseRTT_delta_cwnd_lower"].append(
                one_component_large_cwnd_l - previous_forced_cwnd
            )
            detailed_symbolic_cwnd_info["large_inverseRTT_delta_cwnd_upper"].append(
                one_component_large_cwnd_u - previous_forced_cwnd
            )
    else:
        detailed_symbolic_cwnd_info = None

    symbolic_cwnd_info = {
        "small_inverseRTT_delta_cwnd_lower": small_inverseRTT_forced_cwnd_lower
        - previous_forced_cwnd,
        "small_inverseRTT_delta_cwnd_upper": small_inverseRTT_forced_cwnd_upper
        - previous_forced_cwnd,
        "large_inverseRTT_delta_cwnd_lower": large_inverseRTT_forced_cwnd_lower
        - previous_forced_cwnd,
        "large_inverseRTT_delta_cwnd_upper": large_inverseRTT_forced_cwnd_upper
        - previous_forced_cwnd,
    }

    # Option 1: Only restrict the area where the concrete alpha is in.
    # Option 2: Restrict all the area that the concrete alpha can be in.
    # TODO(chenxiyang): Add Option 1. The code below is for option 2.
    if constraints_id == SAFETY_CONSTRAINTS_ID:
        # large queuing delay
        small_inverseRTT_reward = calculate_safety_reward(
            small_inverseRTT_forced_cwnd_lower,
            small_inverseRTT_forced_cwnd_upper,
            previous_forced_cwnd,
            -x2,
            0,
        )
        # small queuing delay
        large_inverseRTT_reward = calculate_safety_reward(
            large_inverseRTT_forced_cwnd_lower,
            large_inverseRTT_forced_cwnd_upper,
            previous_forced_cwnd,
            0,
            x2,
        )
        r_certified = (small_inverseRTT_reward + large_inverseRTT_reward) / 2.0
    elif constraints_id == LOSS_CONSTRAINTS_ID or constraints_id == LOSS_CONSTRAINTS_LIVENESS_ID:
        small_inverseRTT_reward = calculate_safety_reward(
            small_inverseRTT_forced_cwnd_lower,
            small_inverseRTT_forced_cwnd_upper,
            previous_forced_cwnd,
            -x2,
            0,
        )
        large_inverseRTT_reward = calculate_safety_reward(
            large_inverseRTT_forced_cwnd_lower,
            large_inverseRTT_forced_cwnd_upper,
            previous_forced_cwnd,
            -x2,
            0,
        )
        r_certified = (small_inverseRTT_reward + large_inverseRTT_reward) / 2.0
    elif constraints_id == DEEP_BUFFER_CONSTRAINTS:
        # low queuing delay and past cwnd
        # track previous delta_cwnd_lower, delta_cwnd_upper
        low_queuing_delay_delta_cwnd_lower = large_inverseRTT_forced_cwnd_lower - previous_forced_cwnd
        low_queuing_delay_delta_cwnd_upper = large_inverseRTT_forced_cwnd_upper - previous_forced_cwnd
        # high queuing delay and past cwnd
        high_queuing_delay_delta_cwnd_lower = small_inverseRTT_forced_cwnd_lower - previous_forced_cwnd
        high_queuing_delay_delta_cwnd_upper = small_inverseRTT_forced_cwnd_upper - previous_forced_cwnd
        
        # property 1: low queuing delay and past cwnd delta <= 0 -> current delta cwnd > 0
        # property 2: i). high queuing delay and past cwnd delta >= 0 -> current delta cwnd <= 0
        #             ii). high queuing delay and past cwnd delta <= 0 -> current delta cwnd >= 0
        max_low_delay_cwnd = max(low_delay_delta_cwnd_upper_q) if k != 1 else low_delay_delta_cwnd_upper_q[-1]
        min_high_delay_cwnd = min(high_delay_delta_cwnd_lower_q) if k != 1 else high_delay_delta_cwnd_lower_q[-1]
        max_high_delay_cwnd = max(high_delay_delta_cwnd_upper_q) if k != 1 else high_delay_delta_cwnd_upper_q[-1]
        property1_reward = 1.0
        property2_reward = 1.0
        if max_low_delay_cwnd <= 0:
            property1_reward = calculate_safety_reward(
                large_inverseRTT_forced_cwnd_lower,
                large_inverseRTT_forced_cwnd_upper,
                previous_forced_cwnd,
                0,
                x2,
            )
        if min_high_delay_cwnd >= 0:
            property2_reward = calculate_safety_reward(
                small_inverseRTT_forced_cwnd_lower,
                small_inverseRTT_forced_cwnd_upper,
                previous_forced_cwnd,
                -x2,
                0,
            )
        if max_high_delay_cwnd <= 0:
            property2_reward = calculate_safety_reward(
                small_inverseRTT_forced_cwnd_lower,
                small_inverseRTT_forced_cwnd_upper,
                previous_forced_cwnd,
                0,
                x2,
            )
        
        low_delay_delta_cwnd_lower_q.append(low_queuing_delay_delta_cwnd_lower)
        low_delay_delta_cwnd_upper_q.append(low_queuing_delay_delta_cwnd_upper)
        high_delay_delta_cwnd_lower_q.append(high_queuing_delay_delta_cwnd_lower)
        high_delay_delta_cwnd_upper_q.append(high_queuing_delay_delta_cwnd_upper)
        
        r_certified = (property1_reward + property2_reward) / 2.0
    elif constraints_id == SHALLOW_BUFFER_CONSTRAINTS:
        # min queuing delay and no packet loss (use the same variable large_inverseRTT to represent)
        good_network_delta_cwnd_lower = large_inverseRTT_forced_cwnd_lower - previous_forced_cwnd
        good_network_delta_cwnd_upper = large_inverseRTT_forced_cwnd_upper - previous_forced_cwnd
        # min queuing delay but high packet loss (use the same variable small_inverseRTT to represent)
        bad_network_delta_cwnd_lower = small_inverseRTT_forced_cwnd_lower - previous_forced_cwnd
        bad_network_delta_cwnd_upper = small_inverseRTT_forced_cwnd_upper - previous_forced_cwnd
        
        # property 3: min queuing delay and no packet loss and past cwnd delta <= 0 -> current delta cwnd > 0
        # property 4: min queuing delay and high packet loss and past cwnd delta >= 0 -> current delta cwnd < 0
        max_good_cwnd = max(low_delay_delta_cwnd_upper_q) if k != 1 else low_delay_delta_cwnd_upper_q[-1]
        min_bad_cwnd = min(high_delay_delta_cwnd_lower_q) if k != 1 else high_delay_delta_cwnd_lower_q[-1]
        property3_reward = 1.0
        property4_reward = 1.0
        if max_good_cwnd <= 0:
            property3_reward = calculate_safety_reward(
                large_inverseRTT_forced_cwnd_lower,
                large_inverseRTT_forced_cwnd_upper,
                previous_forced_cwnd,
                0,
                x2,
            )
        if min_bad_cwnd >= 0:
            property4_reward = calculate_safety_reward(
                small_inverseRTT_forced_cwnd_lower,
                small_inverseRTT_forced_cwnd_upper,
                previous_forced_cwnd,
                -x2,
                0,
            )
        
        low_delay_delta_cwnd_lower_q.append(good_network_delta_cwnd_lower)
        low_delay_delta_cwnd_upper_q.append(good_network_delta_cwnd_upper)
        high_delay_delta_cwnd_lower_q.append(bad_network_delta_cwnd_lower)
        high_delay_delta_cwnd_upper_q.append(bad_network_delta_cwnd_upper)
        
        r_certified = (property3_reward + property4_reward) / 2.0
        
    if constraints_id == LOSS_CONSTRAINTS_LIVENESS_ID:
        reward_q.append(r_certified)
        r_certified = min(reward_q) # use the smallest one from the past 3 steps. 
        # As long as there is one step that is not satisfied, we penalize the training.
    
    return r_certified, symbolic_cwnd_info, detailed_symbolic_cwnd_info


def get_robustness_symbolic_constrained_reward(
    state,
    action,
    action_before_noise,
    current_tcp_cwnd,
    previous_action,
    previous_tcp_cwnd,
    threshold,
    x1,
    x2,
    lambda_,
    union_a_symbolic,
    a_symbolic=None,
):
    a_lower = union_a_symbolic.l
    a_upper = union_a_symbolic.u

    current_forced_cwnd = get_forced_cwnd_from_alpha_and_tcp_cwnd(
        action_before_noise, current_tcp_cwnd
    )
    cwnd_lower, cwnd_upper = get_symbolic_forced_cwnd(
        a_lower, a_upper, current_tcp_cwnd
    )
    # Using x2 here.
    # Get the upper and lower of the symbolic cwnd,
    # Get the old cwnd?
    # distance from [cwnd_lower, cwnd_upper] to [current_cwnd - x2, current_cwnd + x2]
    # r_certified = calculate_distance(
    #     cwnd_lower, cwnd_upper, current_forced_cwnd - x2, current_forced_cwnd + x2
    # )
    x2_percent = x2 / 100.0
    r_certified = calculate_distance(
        cwnd_lower,
        cwnd_upper,
        (1 - x2_percent) * current_forced_cwnd,
        (1 + x2_percent) * current_forced_cwnd,
    )
    # print(f"certified_cwnd l, r: {cwnd_lower, cwnd_upper}")
    # print(f"current_cwnd noised range: {current_forced_cwnd - x2, current_forced_cwnd + x2}")
    # print(f"action: {action}; action_before_noise: {action_before_noise}")
    # print(f"r_certified: {r_certified}")
    symbolic_cwnd_info = {
        "l": cwnd_lower,
        "r": cwnd_upper,
    }
    if a_symbolic:
        detailed_symbolic_cwnd_info = {
            "l": [],
            "u": [],
        }
        for one_component_l, one_component_u in zip(a_symbolic.l, a_symbolic.u):
            one_component_cwnd_l, one_component_cwnd_u = get_symbolic_forced_cwnd(
                one_component_l, one_component_u, current_tcp_cwnd
            )
            detailed_symbolic_cwnd_info["l"].append(one_component_cwnd_l)
            detailed_symbolic_cwnd_info["u"].append(one_component_cwnd_u)
    else:
        detailed_symbolic_cwnd_info = None
    
    return r_certified, symbolic_cwnd_info, detailed_symbolic_cwnd_info
    

def get_symbolic_constrained_reward(
    state,
    action,
    action_before_noise,
    current_tcp_cwnd,
    previous_action,
    previous_tcp_cwnd,
    constraints_id,
    threshold,
    x1,
    x2,
    lambda_,
    union_a_symbolic,
    a_symbolic=None,
    k=1,
):
    if constraints_id in {
            SAFETY_CONSTRAINTS_ID,
            LOSS_CONSTRAINTS_ID,
            LOSS_CONSTRAINTS_LIVENESS_ID,
            DEEP_BUFFER_CONSTRAINTS,
            SHALLOW_BUFFER_CONSTRAINTS,
        }:
        r_certified, symbolic_cwnd_info, detailed_symbolic_cwnd_info = get_perf_symbolic_constrained_reward(
            state=state,
            action=action,
            action_before_noise=action_before_noise,
            current_tcp_cwnd=current_tcp_cwnd,
            previous_action=previous_action,
            previous_tcp_cwnd=previous_tcp_cwnd,
            threshold=threshold,
            x1=x1,
            x2=x2,
            lambda_=lambda_,
            constraints_id=constraints_id,
            union_a_symbolic=union_a_symbolic,
            a_symbolic=a_symbolic,
            k=k,
        )
    elif constraints_id == ROBUSTNESS_CONSTRAINTS_ID:
        r_certified, symbolic_cwnd_info, detailed_symbolic_cwnd_info = get_robustness_symbolic_constrained_reward(
            state=state,
            action=action,
            action_before_noise=action_before_noise,
            current_tcp_cwnd=current_tcp_cwnd,
            previous_action=previous_action,
            previous_tcp_cwnd=previous_tcp_cwnd,
            threshold=threshold,
            x1=x1,
            x2=x2,
            lambda_=lambda_,
            union_a_symbolic=union_a_symbolic,
            a_symbolic=a_symbolic,
        )
    elif constraints_id == PERF_ROBUSTNESS_CONSTRAINTS_ID:
        r_certified_perf, symbolic_cwnd_info_perf, detailed_symbolic_cwnd_info_perf = get_perf_symbolic_constrained_reward(
            state=state,
            action=action,
            action_before_noise=action_before_noise,
            current_tcp_cwnd=current_tcp_cwnd,
            previous_action=previous_action,
            previous_tcp_cwnd=previous_tcp_cwnd,
            threshold=threshold,
            x1=x1,
            x2=x2,
            lambda_=lambda_,
            constraints_id=SAFETY_CONSTRAINTS_ID,
            union_a_symbolic=union_a_symbolic['perf'],
            a_symbolic=a_symbolic['perf'] if a_symbolic else None,
        )
        r_certified_robust, symbolic_cwnd_info_robust, detailed_symbolic_cwnd_info_robust = get_robustness_symbolic_constrained_reward(
            state=state,
            action=action,
            action_before_noise=action_before_noise,
            current_tcp_cwnd=current_tcp_cwnd,
            previous_action=previous_action,
            previous_tcp_cwnd=previous_tcp_cwnd,
            threshold=threshold,
            x1=x1,
            x2=x2,
            lambda_=lambda_,
            union_a_symbolic=union_a_symbolic['robustness'],
            a_symbolic=a_symbolic['robustness'] if a_symbolic else None,
        )
        # TODO: we can adjust the combination here.
        r_certified = (r_certified_perf + r_certified_robust) / 2.0
        symbolic_cwnd_info = {
            "small_inverseRTT_delta_cwnd_lower": symbolic_cwnd_info_perf["small_inverseRTT_delta_cwnd_lower"],
            "small_inverseRTT_delta_cwnd_upper": symbolic_cwnd_info_perf["small_inverseRTT_delta_cwnd_upper"],
            "large_inverseRTT_delta_cwnd_lower": symbolic_cwnd_info_perf["large_inverseRTT_delta_cwnd_lower"],
            "large_inverseRTT_delta_cwnd_upper": symbolic_cwnd_info_perf["large_inverseRTT_delta_cwnd_upper"],
            "l": symbolic_cwnd_info_robust["l"],
            "r": symbolic_cwnd_info_robust["r"],
        }
        if a_symbolic:
            detailed_symbolic_cwnd_info = {
                "small_inverseRTT_delta_cwnd_lower": detailed_symbolic_cwnd_info_perf["small_inverseRTT_delta_cwnd_lower"],
                "small_inverseRTT_delta_cwnd_upper": detailed_symbolic_cwnd_info_perf["small_inverseRTT_delta_cwnd_upper"],
                "large_inverseRTT_delta_cwnd_lower": detailed_symbolic_cwnd_info_perf["large_inverseRTT_delta_cwnd_lower"],
                "large_inverseRTT_delta_cwnd_upper": detailed_symbolic_cwnd_info_perf["large_inverseRTT_delta_cwnd_upper"],
                "l": detailed_symbolic_cwnd_info_robust["l"],
                "u": detailed_symbolic_cwnd_info_robust["u"],
            }
        else:
            detailed_symbolic_cwnd_info = None
    else:
        raise ValueError(f"Constraint {constraints_id} not implemented.")

    return r_certified, symbolic_cwnd_info, detailed_symbolic_cwnd_info


def get_raw_and_constraint_reward(
    state,
    action,
    action_before_noise,
    raw_reward,
    current_tcp_cwnd,
    constraints_id,
    reward_mode,
    threshold,
    x1,
    x2,
    lambda_,
    union_a_symbolic,
    past_states,
    past_actions,
    past_tcp_cwnds,
    k,
    a_symbolic=None,
):
    """
    Extract the original reward, constrained reward, and the combined reward by constraints_id
    and reward_mode.

    Args:
        state: state
        action: action
        action_before_noise: action_before_noise
        raw_reward: original Orca defined reward
        current_tcp_cwnd: current tcp cwnd
        previous_action: previous action
        previous_tcp_cwnd: previous tcp cwnd
        constraints_id: id of the constraint to use
        reward_mode: string. whether to combine the reward or not.
        threshold: threshold value for the constraint
        x1: the first changing point for the reward
        x2: the second changing point for the reward
        lambda_: the weight for the reward and the alpha_delta_reward
        k: the number of steps to consider for the long horizon constraints
        a_symbolic: The alpha of each symbolic components.

    Returns:
        raw_reward: the original reward value from Orca.
        constrained_reward: the reward about constraint satisfaction.
        overall_reward: the combined reward.
    """
    if past_states:
        (
            empirical_constraint_reward,
            heuristic_constraints_info,
        ) = get_empirical_constrained_reward(
            state=state,
            action=action,
            current_tcp_cwnd=current_tcp_cwnd,
            previous_action=past_actions[-1],
            previous_tcp_cwnd=past_tcp_cwnds[-1],
            constraints_id=constraints_id,
            threshold=threshold,
            x1=x1,
            x2=x2,
            lambda_=lambda_,
            past_states=past_states,
            past_actions=past_actions,
            past_tcp_cwnds=past_tcp_cwnds,
            k=k,
        )
        (
            symbolic_constraint_reward,
            symbolic_cwnd_info,
            detailed_symbolic_cwnd_info,
        ) = get_symbolic_constrained_reward(
            state=state,
            action=action,
            action_before_noise=action_before_noise,
            current_tcp_cwnd=current_tcp_cwnd,
            previous_action=past_actions[-1],
            previous_tcp_cwnd=past_tcp_cwnds[-1],
            constraints_id=constraints_id,
            threshold=threshold,
            x1=x1,
            x2=x2,
            lambda_=lambda_,
            union_a_symbolic=union_a_symbolic,
            a_symbolic=a_symbolic,
            k=k,
        )
    else:
        empirical_constraint_reward = 0.0
        symbolic_constraint_reward = 0.0
        if constraints_id in {
            SAFETY_CONSTRAINTS_ID,
            LOSS_CONSTRAINTS_ID,
            LOSS_CONSTRAINTS_LIVENESS_ID,
            DEEP_BUFFER_CONSTRAINTS,
            SHALLOW_BUFFER_CONSTRAINTS,
        }:
            symbolic_cwnd_info = {
                "small_inverseRTT_delta_cwnd_lower": 0.0,
                "small_inverseRTT_delta_cwnd_upper": 0.0,
                "large_inverseRTT_delta_cwnd_lower": 0.0,
                "large_inverseRTT_delta_cwnd_upper": 0.0,
            }
            detailed_symbolic_cwnd_info = {
                "small_inverseRTT_delta_cwnd_lower": [],
                "small_inverseRTT_delta_cwnd_upper": [],
                "large_inverseRTT_delta_cwnd_lower": [],
                "large_inverseRTT_delta_cwnd_upper": [],
            }
        elif constraints_id == ROBUSTNESS_CONSTRAINTS_ID:
            symbolic_cwnd_info = {
                "l": 0.0,
                "r": 0.0,
            }
            detailed_symbolic_cwnd_info = {
                "l": [],
                "u": [],
            }
        elif constraints_id == PERF_ROBUSTNESS_CONSTRAINTS_ID:
            symbolic_cwnd_info = {
                "small_inverseRTT_delta_cwnd_lower": 0.0,
                "small_inverseRTT_delta_cwnd_upper": 0.0,
                "large_inverseRTT_delta_cwnd_lower": 0.0,
                "large_inverseRTT_delta_cwnd_upper": 0.0,
                "l": 0.0,
                "r": 0.0,
            }
            detailed_symbolic_cwnd_info = {
                "small_inverseRTT_delta_cwnd_lower": [],
                "small_inverseRTT_delta_cwnd_upper": [],
                "large_inverseRTT_delta_cwnd_lower": [],
                "large_inverseRTT_delta_cwnd_upper": [],
                "l": [],
                "u": [],
            }
        else:
            raise ValueError(f"Constraint {constraints_id} not implemented.")
        heuristic_constraints_info = {
            "latency_signal": "mid",
            "delta_cwnd": 0.0,
        }
    if reward_mode == NO_CONSTRAINED:
        overall_reward = raw_reward
    elif reward_mode == ONLY_EMPIRICAL_CONSTRAINED:
        overall_reward = empirical_constraint_reward
    elif reward_mode == ONLY_SYMBOLIC_CONSTRAINED:
        overall_reward = symbolic_constraint_reward
    elif reward_mode == EMPIRICAL_CONSTRAINED:
        overall_reward = (
            1 - lambda_
        ) * raw_reward + lambda_ * empirical_constraint_reward
    elif reward_mode == SYMBOLIC_CONSTRAINED:
        overall_reward = (
            1 - lambda_
        ) * raw_reward + lambda_ * symbolic_constraint_reward
    elif reward_mode == EMPIRICAL_AND_SYMBOLIC_CONSTRAINED:
        overall_reward = (
            1 - lambda_
        ) * empirical_constraint_reward + lambda_ * symbolic_constraint_reward
    elif reward_mode == RAW_AND_SYMBOLIC_CONSTRAINED:
        overall_reward = (
            1 - lambda_
        ) * raw_reward + lambda_ * symbolic_constraint_reward
    # print(f"raw_reward: {raw_reward}, constraint_reward: {constraint_reward}, overall_reward: {overall_reward}")
    return (
        raw_reward,
        empirical_constraint_reward,
        symbolic_constraint_reward,
        overall_reward,
        symbolic_cwnd_info,
        detailed_symbolic_cwnd_info,
        heuristic_constraints_info,
    )
