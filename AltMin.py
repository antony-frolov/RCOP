import numpy as np
import time
from scipy.optimize import minimize
np.random.seed(0)


def mse(u, v, obj_matrix, rank, known_entries):
    x = u.reshape((obj_matrix.shape[0], rank)) @ v.reshape((rank, obj_matrix.shape[1]))
    return np.sum(((x - obj_matrix) * known_entries) ** 2) / (np.sum(known_entries))


def alt(func, rank, k, obj_matrix, known_entries):
    def f(u, v):
        return func(u, v, obj_matrix, rank, known_entries)
    m = obj_matrix.shape[0]
    n = obj_matrix.shape[1]
    u = np.random.randint(0, 10, m * rank)
    v = np.random.randint(0, 10, rank * n)
    time_arr = []
    mse_arr = []
    for i in range(k):
        start_time = time.time()
        v = minimize(lambda x: f(u, x), v).x
        u = minimize(lambda x: f(x, v), u).x
        iter_time = time.time() - start_time
        time_arr.append(iter_time)
        print("k =", i + 1)
        print("time_k =", iter_time)
        mse_k = f(u, v)
        mse_arr.append(mse_k)
        print("mse_k =", mse_k)
    return u.reshape((obj_matrix.shape[0], rank)) @ v.reshape((rank, obj_matrix.shape[1])), time_arr, mse_arr


m = 30
n = 45
p = 0.8
obj_matrix = np.random.randint(0, 10, m * n).reshape((m, n))
known_entries = np.random.choice(a=[True, False], size=(m, n), p=[p, 1-p])

res, time_arr, mse_arr = alt(func=mse, rank=20, k=5, obj_matrix=obj_matrix, known_entries=known_entries)
print(np.mean((res - obj_matrix) ** 2))
print(np.linalg.matrix_rank(res))
print(res - obj_matrix)

import matplotlib.pyplot as plt


plt.plot(list(range(1, len(mse_arr) + 1)), mse_arr)
plt.xlabel("$k$")
plt.ylabel("MSE")
plt.show()
plt.show()
plt.plot(list(range(1, len(time_arr) + 1)), time_arr)
plt.xlabel("$k$")
plt.ylabel("Время")
plt.show()
plt.show()
print(np.mean(time_arr))
