import numpy as np

# Stochastic Adjacency Matrix
M = np.array([[0, 0, 0, 1, 0],
              [1/2, 0, 0, 0, 0],
              [0, 1/2, 0, 0, 0],
              [1/2, 1/2, 0, 0, 1],
              [0, 0, 0, 0, 0]])

# Terminate Offset
epsilon = 0.00001

# Dampening Factor
beta = 0.85

print("M = \n", M, "\n")

n = len(M)
np_ones = np.ones(n)
A = (1 - beta) / n * np_ones * np.transpose(np_ones) + M * beta
print("A = \n", A, "\n")


def pagerank(M, epsilon):
    epoch = 0
    r = np.ones(len(M)) / len(M)
    if(epoch == 0):
        print('Original rank vector', r, '\n')
    loop = True
    while loop:
        r_old = r
        r = M.dot(r)
        loop = False
        epoch += 1  # update epoch counter
        for diff in np.abs(r - r_old):
            if diff >= epsilon:
                loop = True
                break
    return r, epoch

float_formatter = "{:.5f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})

m_r, m_e = pagerank(M, epsilon)
print('Rank vector for matrix M:', m_r)
print('Epochs for matrix M:', m_e)

print('')

a_r, a_e = pagerank(A, epsilon)
print('Rank vector for matrix A:', a_r)
print('Epochs for matrix A:', a_e)
