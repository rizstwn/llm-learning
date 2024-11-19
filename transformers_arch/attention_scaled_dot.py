import numpy as np
import scipy

def attention(Q,K,V):
    scores = np.matmul(Q,K.T)
    weights = scipy.special.softmax(scores / K.shape[1] ** 0.5, axis=1)
    attention = np.matmul(weights, V)
    return attention