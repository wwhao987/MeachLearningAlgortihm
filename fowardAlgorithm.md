### FowardAlgorithm

##### formula

The Forward Algorithm computes a forward probability matrix `F`, with dimensions (number of states) x (number of observations). The entry `F[s, t]` represents the probability of being in state `s` after the first `t` observations, given the model's parameters.

```python
constructure step
"""
Here is the algorithm in words:

1 Create a probability matrix F of size N x T, 
where N is the number of states and T is the number of time steps (i.e., length of observation sequence).

2 Initialize the first column (t=0) of this matrix with the start probabilities of each state multiplied by the emission probability of the observed symbol at time t=0.

3 For each time step from t=1 to T-1:

 For each state s:
Calculate the total probability of being in state s at time t by summing over the probabilities of being in each state s' at time t-1 and transitioning from s' to s, and then emitting the observed symbol at time t.
This is done using the forward formula: F[s, t] = sum(F[s', t-1] * a[s', s] * b[s, O[t]) for all s', where a[s', s] is the transition probability from state s' to s, b[s, O[t]] is the emission probability of emitting symbol O[t] at state s, and O[t] is the observed symbol at time t.
"""
# 1 The first column of `F` is initialized as follows:
F[s, 0] = start_probability[s] * emission_probability[s, observation[0]]
# 2 Recursion: Then for each subsequent observation from t=1 to t=T-1 (where T is the number of observations), the entries of F are updated as follows:
F[s, t] = Σ(F[s', t-1] * transition_probability[s', s] * emission_probability[s, observation[t]])
# 3The sum is over all states s'.
# 3Termination: Finally, the probability of the entire observed sequence given the model is computed by summing the probabilities in the last column of the forward probability matrix:
P(observation|model) = Σ(F[s, T-1])

```

```python
import numpy as np

def forward(obs_seq, states, start_prob, trans_prob, emit_prob):
    N = len(states)
    T = len(obs_seq)
    
    F = np.zeros((N, T))
    for s in range(N):
        F[s, 0] = start_prob[s] * emit_prob[s, obs_seq[0]]
    
    for t in range(1, T):
        for s in range(N):
            F[s, t] = np.sum([F[s_prev, t-1] * trans_prob[s_prev, s] * emit_prob[s, obs_seq[t]] for s_prev in range(N)])

    return np.sum(F[:, -1])

# example usage:

states = ['Rainy', 'Sunny']
obs_seq = ['walk', 'shop', 'clean']
start_prob = np.array([0.6, 0.4])
trans_prob = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])
emit_prob = np.array([
    [0.1, 0.4, 0.5],
    [0.6, 0.3, 0.1]
])

obs_map = {'walk': 0, 'shop': 1, 'clean': 2}
obs_seq = [obs_map[o] for o in obs_seq]

prob_seq = forward(obs_seq, states, start_prob, trans_prob, emit_prob)
print('Probability of observing the sequence given the model: ', prob_seq)

```



#  Backward Algorithm

```python

# Initialization: Create a probability matrix B of size N x T, where N is the number of states and T is the number of time steps (i.e., length of observation sequence). Initialize the last column (t=T-1) of this matrix with 1, as the probability of ending the sequence after the last observation is 1 for all states.

B[s, T-1] = 1  for all s
# Recursion: Then for each previous observation from t=T-2 to t=0, the entries of B are updated as follows:
B[s, t] = Σ (B[s', t+1] * transition_probability[s, s'] * emission_probability[s', observation[t+1]])  for all s'
# Termination: Finally, the probability of the entire observed sequence given the model is computed by summing the product of the initial state probability, the emission probability of the first observation, and the backward probability for all states:

P(observation|model) = Σ (start_probability[s] * emission_probability[s, observation[0]] * B[s, 0])  for all s
This algorithm is similar to the forward algorithm, but it works in reverse, starting at the end of the sequence and working back to the beginning.
```

```python
import numpy as np

def backward(obs_seq, states, start_prob, trans_prob, emit_prob):
    N = len(states)
    T = len(obs_seq)
    
    B = np.zeros((N, T))
    B[:, -1] = 1  # Initialization: set last column to 1
    
    for t in range(T-2, -1, -1):  # Recursion: from T-2 down to 0
        for s in range(N):
            B[s, t] = np.sum([B[s_next, t+1] * trans_prob[s, s_next] * emit_prob[s_next, obs_seq[t+1]] for s_next in range(N)])
            
    # Termination
    prob_seq = np.sum([start_prob[s] * emit_prob[s, obs_seq[0]] * B[s, 0] for s in range(N)])
    
    return prob_seq

# example usage:

states = ['Rainy', 'Sunny']
obs_seq = ['walk', 'shop', 'clean']
start_prob = np.array([0.6, 0.4])
trans_prob = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])
emit_prob = np.array([
    [0.1, 0.4, 0.5],
    [0.6, 0.3, 0.1]
])

obs_map = {'walk': 0, 'shop': 1, 'clean': 2}
obs_seq = [obs_map[o] for o in obs_seq]

prob_seq = backward(obs_seq, states, start_prob, trans_prob, emit_prob)
print('Probability of observing the sequence given the model: ', prob_seq)

```



