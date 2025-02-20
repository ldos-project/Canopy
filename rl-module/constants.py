
'''
We calculate the latency gradient at time step t by the difference of 
the average latency of [t-GRADIENT_STEPS-1, t-1] and
[t-GRADIENT_STEPS, t]. To simply, the latency gradient = latency_t - latency_{t-GRADIENT_STEPS-1}
'''
GRADIENT_STEPS = 9
assert(GRADIENT_STEPS <= 9) 

from constrained_reward import(
    SAFETY_CONSTRAINTS_ID,
    ROBUSTNESS_CONSTRAINTS_ID,
    LOSS_CONSTRAINTS_ID,
    LOSS_CONSTRAINTS_LIVENESS_ID,
    DEEP_BUFFER_CONSTRAINTS,
    SHALLOW_BUFFER_CONSTRAINTS,
)

class SymbolicTransitionState():
    def __init__(self, inverse_rtt_lower, inverse_rtt_upper, inverse_gradient_lower, inverse_gradient_upper):
        self.inverse_rtt_lower = inverse_rtt_lower
        self.inverse_rtt_upper = inverse_rtt_upper
        self.inverse_gradient_lower = inverse_gradient_lower
        self.inverse_gradient_upper = inverse_gradient_upper

class SymbolicComponent_c_delta():
    def __init__(self,
                 constraints_id,
                 small_c=None,
                 small_delta=None,
                 large_c=None,
                 large_delta=None,
                 c=None,
                 delta=None
                 ):
        # small is the component for small inverseRTT case
        # large is the component for large inverseRTT case
        if constraints_id in {
                SAFETY_CONSTRAINTS_ID,
                LOSS_CONSTRAINTS_ID,
                LOSS_CONSTRAINTS_LIVENESS_ID,
                DEEP_BUFFER_CONSTRAINTS,
                SHALLOW_BUFFER_CONSTRAINTS,
            }:
            self.small_c = small_c
            self.small_delta = small_delta
            self.large_c = large_c
            self.large_delta = large_delta
        elif constraints_id == ROBUSTNESS_CONSTRAINTS_ID:
            self.c = c
            self.delta = delta
        else:
            raise ValueError(f"Invalid constraints_id: {constraints_id}")

class SymbolicComponent_bound():
    def __init__(self,
                 constraints_id,
                 small_l=None,
                 small_u=None,
                 large_l=None,
                 large_u=None,
                 l=None,
                 u=None,
                 ):
        if constraints_id in {
                SAFETY_CONSTRAINTS_ID,
                LOSS_CONSTRAINTS_ID,
                LOSS_CONSTRAINTS_LIVENESS_ID,
                DEEP_BUFFER_CONSTRAINTS,
                SHALLOW_BUFFER_CONSTRAINTS,
            }:
            self.small_l = small_l
            self.small_u = small_u
            self.large_l = large_l
            self.large_u = large_u
        elif constraints_id == ROBUSTNESS_CONSTRAINTS_ID:
            self.l = l
            self.u = u
        else:
            raise ValueError(f"Invalid constraints_id: {constraints_id}")
    
