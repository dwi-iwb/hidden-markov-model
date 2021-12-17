import pandas as pd
import numpy as np
import difflib as dl
import matplotlib.pyplot as plt

# Load and lean Data

data = pd.read_csv("Sequence_case2.txt", sep="/n", header=None, engine="python" )
data = data.drop([0,2]).T
data = data.set_axis(['prot_seq', 'transmem_reg'], axis='columns')
data['prot_seq'][0] = list(data['prot_seq'][0])
data['transmem_reg'][0] = list(data['transmem_reg'][0])
data = data.explode(['prot_seq', 'transmem_reg'])
data = data.astype({'transmem_reg': 'int32'})

# Model Parameters

def transition_matrix(Q):
    transitions = np.zeros((2,2)) # TODO get dimension from Q
    for i in range(len(Q)-1): transitions[Q[i]][Q[i+1]] += 1 
    transitions /= np.sum(transitions, axis=1).reshape(-1,1) # divide every row by its sum
    return(transitions)

# structure
S = { 0, 1 } # #N states 
V = { 'C': 0, 'N': 1, 'H': 2 }   # observation vocabulary M
data['prot_seq'] = data['prot_seq'].replace(V) # encode categories
# observations
O = data['prot_seq'].values # #T observations
Q = data['transmem_reg'].values # hidden state sequence / ground truth
# a)
# probabilities: λ
Pi = np.array((1, 0)) # Estimated by looking at the first data points
A = transition_matrix(Q) # transition probabilitie matrix
distribution = data.groupby(['transmem_reg', 'prot_seq']).size()
B = np.array((tuple(distribution[0]/distribution[0].sum()),
              tuple(distribution[1]/distribution[1].sum()))) # emission probabilitie matrix / observation likelihood 

print(A, " A")
print(Pi, 'π')
print(B, "B")


# Viterbi Algorithm
# b)
# http://www.adeveloperdiary.com/data-science/machine-learning/implement-viterbi-algorithm-in-hidden-markov-model-using-python-and-r/
def viterbi(O, A, B, Pi):
    T = O.shape[0]
    N = A.shape[0]
    print(type(A[0]))
 
    omega = np.zeros((T, N))
    # Init matrix (t=0)
    omega[0, :] = np.log(Pi * B[:, O[0]])
 
    prev = np.zeros((T - 1, N))
 
    for t in range(1, T):
        for s in range(N):
            # Same as Forward Probability
            probability = omega[t-1] + np.log(A[:, s]) + np.log(B[s, O[t]])
            # This is our most probable state given previous state at time t (1)
            prev[t - 1, s] = np.argmax(probability)
            # This is the probability of the most probable state (2)
            omega[t, s] = np.max(probability)
 
    # Path Array
    S = np.zeros(T)
 
    # Find the most probable last hidden state
    last_state = np.argmax(omega[T - 1, :])
 
    S[0] = last_state
 
    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1
 
    # Flip the path array since we were backtracking
    S = np.flip(S, axis=0)
 
    # Convert numeric values to actual hidden states
    result = []
    for s in S:
        result.append(int(s))
 
    return result


model_prediction = viterbi(O, A, B, Pi)

for i in range(len(model_prediction)):
    print(list(Q)[i],model_prediction[i])

match_ratio = len([i for i, j in zip(list(Q), model_prediction) if i == j])/len(model_prediction)
print(match_ratio)


