import numpy as np

def viterbi(init_dist, transition, emission, observations):
    """
    Given: HMM with K states, observation space of size L, observation sequence after T time steps

    Input:
        init_dist: initial state distribution,
        transition: matrix with transition probabilities for each state, shape: (K,K)
        emission: matrix with emission probabilities for each state, shape: (K, L)
        observations: sequence of emitted observations, shape: (T)

    Output:
        (viterbi_table, backtrace_table, state_sequence), where:
        viterbi_table: filled table with probabilities
        backtrace_table: table to backtrace which state sequence gave most likely sequence
        state_sequence: sequence of states that is most likely
    """
    
    # Retrieve state space, sequence length
    t = len(observations)
    k = transition.shape[0]
    
    # Create arrays to store probabilities, backtrace
    viterbi = np.zeros((k,t), dtype=np.float64) # viterbi table
    backtrace = np.zeros((k,t), dtype=np.int) # backtrace table
    path = np.zeros(t, dtype=np.int) # best path recovered by backtrace

    # Initialize table with initial distribution and first observation
    #viterbi[:,0] = init_dist * emission[:, observations[0]]
    #backtrace[:,0] = 0

    # Use recurrence relation to iterate over all observations
    #for i in range(1,t):
    #    viterbi[:, i] = np.max(viterbi[:, i-1] * transition.T * emission[np.newaxis, :, observations[i]].T, 1)
    #    backtrace[:, i] = np.argmax(viterbi[:, i-1] * transition.T, 1)
   
    # Use recurrence relation to iterate over all observations
    for i, obs in enumerate(observations):
        # Initialize table with initial distribution and first observation (base case) 
        if i == 0:
            viterbi[:,0] = init_dist * emission[:, observations[0]]
            backtrace[:,0] = 0
        # Proceed for all further observations
        else:
            viterbi[:, i] = np.max(viterbi[:, i-1] * transition.T * emission[np.newaxis, :, observations[i]].T, 1)
            backtrace[:, i] = np.argmax(viterbi[:, i-1] * transition.T, 1)

    # Finally obtain most probable sequence from backtrace table
    path[-1] = np.argmax(viterbi[:, t-1])
    for i in reversed(range(1,t)):
        path[i-1] = backtrace[path[i], i]

    return viterbi, backtrace, path

def forward(init_dist, transition, emission, observations):
    """
    Given: HMM with K states, observation space of size L, observation sequence after T time steps

    Input:
        init_dist: initial state distribution,
        transition: matrix with transition probabilities for each state, shape: (K,K)
        emission: matrix with emission probabilities for each state, shape: (K, L)
        observations: sequence of emitted observations, shape: (T)
    """
    
    # Retrieve state space, sequence length
    t = len(observations)
    k = transition.shape[0]

    # Initialize forward table
    forward = np.zeros((k,t))
            
    # Iterate over all observations with recurrence relation
    for i, obs in enumerate(observations):
        # Initialize table with initial distribution and first observation (base case)
        if i == 0:
            forward[:,0] = init_dist * emission[:, observations[0]]
        # Iterate over all states in state space and fill table
        else:
            for j in range(transition.shape[0]):
                forward[j, i] = forward[:,i-1].dot(transition[:, j].T) * emission[j, observations[i]]
    
    p_fwd = np.sum(forward, axis=0)[-1]
    
    return forward, p_fwd

def backward(init_dist, transition, emission, observations):
    """
    Given: HMM with K states, observation space of size L, observation sequence after T time steps

    Input:
        init_dist: initial state distribution,
        transition: matrix with transition probabilities for each state, shape: (K,K)
        emission: matrix with emission probabilities for each state, shape: (K, L)
        observations: sequence of emitted observations, shape: (T)
    """
    
    # Retrieve state space, sequence length
    t = len(observations)
    k = transition.shape[0]

    # Initialize backward table
    backward = np.zeros((k,t))

    # Set final states to 1
    for i in range(k):
        backward[i, t-1] = 1

    for i in reversed(range(t-1)):
        for j in range(transition.shape[0]):
            backward[j, i] = (backward[:,i+1] * emission[:, observations[i+1]]).dot(transition[j,:])
   
    p_bkw = sum(backward[j,0] * init_dist[j] * emission[j, observations[0]] for j in range(k))

    return backward, p_bkw

def forward_backward(init_dist, transition, emission, observations):
    fwd, p_fwd = forward(init_dist, transition, emission, observations)
    bkw, p_bkw = backward(init_dist, transition, emission, observations)

    # Retrieve state space, sequence length
    t = len(observations)
    k = transition.shape[0]

    # Initialize store of posterior probabilities
    posterior = np.zeros((k,t))

    # Compute posterior for each observation
    for i, obs in enumerate(observations):
        for j in range(k):
            #p_fwd = np.sum(fwd, axis=0)[-1]
            posterior[j,i] = fwd[j,i] * bkw[j,i] / p_fwd

    seq = np.argmax(posterior, 0)

    return posterior, seq, p_fwd

init = np.array([0.5, 0.5])
obs = np.array([0,0,0,1,0,0]) # first 6 obs
seq = "HHHTHHTHHHHTTHHHHHHTHHTHHTHHHTHHTTHHHTHHHHHT"
dictionary = {"H": 0, "T": 1} # we use this to convert sequence of observations to numerical observations
reverse_dictionary = {0: "F", 1: "L"} # state 0 corresponds to fair, state 1 corresponds to loaded
obs = np.array([dictionary[char] for char in seq])
transition = np.array([[0.8,0.2],[0.3,0.7]])
emission = np.array([[0.5,0.5], [0.9,0.1]])

print("Observations:", obs)
print("")

print("---------- Viterbi algorithm -----------")
v_table, backtrace, seq = viterbi(init, transition, emission, obs)
print("Most probable state sequence:", [reverse_dictionary[c] for c in seq])
print("Viterbi table (first row fair, second row loaded):")
print(v_table)

print("---------- Posterior decoding ----------")
posterior, seq, p_x = forward_backward(init, transition, emission, obs)
print("Most likely sequence by posterior decoding:", [reverse_dictionary[c] for c in seq])
print("P(X):", p_x)
print("Posterior table (first row fair, second row loaded):")
print(posterior)
