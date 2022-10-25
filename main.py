import numpy as np

def sigmoid (x, der = False):
    if der:
        return x*(1-x)
    return 1 / (1+np.exp(-x))

x = np.array ([[1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
        [-1, -1, -1]])
y = np.array([[0.5, 1, 1.5, 2, 2.5]]).T
np.random.seed(1)
syn0=2 * np.random.random((3, 1))-1
l1=[]
for iter in range (10000):
    l0 = x
    l1= sigmoid(np.dot(l0, syn0))

    l1_error = y - l1

    l1_delta = l1_error * sigmoid(l1, True)
    syn0 += np.dot(l0.T, l1_delta)

print("Выходные данные после тренировки: ")
print(l1)