import numpy as np
import scipy

# Vector representations of words
word_1 = np.array([1, 0, 0])
word_2 = np.array([0, 1, 0])
word_3 = np.array([1, 1, 0])
word_4 = np.array([0, 0, 1])
words = np.array([word_1, word_2, word_3, word_4])

np.random.seed(42)
W_Q = np.random.randint(3, size=(3, 3))
W_K = np.random.randint(3, size=(3, 3))
W_V = np.random.randint(3, size=(3, 3))

queries = np.matmul(words, W_Q)
keys = np.matmul(words, W_K)
values = np.matmul(words, W_V)

scores = np.matmul(queries, keys.T)
weights = scipy.special.softmax(scores / keys.shape[1] ** 0.5, axis=1)
attention = np.matmul(weights, values)
print(attention)