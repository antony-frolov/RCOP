import numpy as np
import cvxpy as cp
import time
from numpy.linalg import eigh, eigvalsh
np.random.seed(0)

def mse(x, obj_matrix, known_entries):
    return np.sum(((x - obj_matrix) * known_entries) ** 2) / (np.sum(known_entries))


def get_first_k_eig_vec(x, k):
    eig_vals, eig_vecs = eigh(x)
    return eig_vecs[:, :k]


def alg(obj_matrix, known_entries, rank, omega_0=1., t=2., eps_1=0.1, eps_2=0.1, k_max=10):
    m = obj_matrix.shape[0]
    n = obj_matrix.shape[1]

    def f(x_in):
        return mse(x_in, obj_matrix, known_entries)

    start_time = time.time()
    x = cp.Variable((m, n))
    obj = cp.Minimize(cp.sum_squares(x - obj_matrix) / (m * n))
    prob = cp.Problem(obj)
    prob.solve()
    x_0 = x.value
    z_0 = np.transpose(x_0) @ x_0
    aux1 = np.concatenate((np.eye(m), x_0), axis=1)
    aux2 = np.concatenate((np.transpose(x_0), z_0), axis=1)
    aux = np.concatenate((aux1, aux2), axis=0)
    v_k_1 = get_first_k_eig_vec(z_0, n - rank)
    v_k_2 = get_first_k_eig_vec(aux, n)
    x_k_prev = np.zeros((m, n))
    x_k = x_0
    e_k = max(eigvalsh(z_0)[n - rank - 1], eigvalsh(aux)[n - 1])
    omega_k = omega_0
    print("k = 0")
    mse_0 = mse(x_0, obj_matrix, known_entries)
    mse_arr = [mse_0]
    print("mse =", mse_0)
    iter_time = time.time() - start_time
    time_arr = [iter_time]
    print("iter_time =", iter_time)
    e_arr = []
    k = 1
    while (k <= k_max) and (e_k >= eps_1) or (np.abs(f(x_k) - f(x_k_prev) / f(x_k_prev)) >= eps_2):
        start_time = time.time()
        # update prev
        x_k_prev = x_k
        e_k_prev = e_k
        v_k_prev_1 = v_k_1
        v_k_prev_2 = v_k_2
        omega_k_prev = omega_k
        # solve SDP problem
        x = cp.Variable((m, n))
        z = cp.Variable((n, n), PSD=True)
        aux = cp.Variable((m + n, m + n), PSD=True)
        e = cp.Variable()
        obj = cp.Minimize(cp.sum_squares(cp.multiply(x - obj_matrix, known_entries)) / np.sum(known_entries)
                          + omega_k * e)
        constr1 = (e * np.eye(n - rank) - np.transpose(v_k_prev_1) @ z @ v_k_prev_1) >> 0
        constr2 = (aux == (np.eye(m + n, m) @ (np.eye(m, m + n) + x @ np.eye(n, m + n, k=m)) +
                           np.eye(m + n, n, k=-m) @ (x.T @ np.eye(m, m + n) + z @ np.eye(n, m + n, k=m))))
        constr3 = (e * np.eye(n) - np.transpose(v_k_prev_2) @ aux @ v_k_prev_2) >> 0
        constr4 = e <= e_k_prev
        prob = cp.Problem(obj, constraints=[constr1, constr2, constr3, constr4])
        prob.solve(solver="MOSEK")
        print("k =", k)
        print("omega_k =", omega_k)
        x_k = x.value
        z_k = z.value
        print("rank z_k =", np.linalg.matrix_rank(z_k, tol=0.005))
        # aux1_k = np.concatenate((np.eye(m), x_k), axis=1)
        # aux2_k = np.concatenate((np.transpose(x_k), z_k), axis=1)
        # aux_k = np.concatenate((aux1_k, aux2_k), axis=0)
        aux_k = aux.value
        print("rank aux_k =", np.linalg.matrix_rank(aux_k, tol=0.005))
        e_k = e.value
        e_arr.append(e_k)
        print("e_k =", e_k)
        # find v_k
        v_k_1 = get_first_k_eig_vec(z_k, n - rank)
        v_k_2 = get_first_k_eig_vec(aux_k, n)
        # update k
        k += 1
        # update omega
        omega_k = omega_k_prev * t
        mse_k = mse(x_k, obj_matrix, known_entries)
        mse_arr.append(mse_k)
        print("mse =", mse_k)
        iter_time = time.time() - start_time
        time_arr.append(iter_time)
        print("iter_time =", iter_time)
    return x_k, time_arr, e_arr

m = 15
n = 20
p = 0.8
known_entries = np.random.choice(a=[True, False], size=(m, n), p=[p, 1-p])
obj_matrix = np.random.randint(0, 10, m * n).reshape((m, n))

print(np.linalg.matrix_rank(obj_matrix))
res, time_arr, e_arr = alg(obj_matrix, known_entries, rank=10, k_max=5, eps_1=0.001, eps_2=1, omega_0=0.1, t=1.5)
print(res)
print(np.linalg.matrix_rank(res, tol=0.05))
print(mse(res, obj_matrix, known_entries))
print(res - obj_matrix)

import matplotlib.pyplot as plt

plt.plot(list(range(1, len(e_arr) + 1)), e_arr)
plt.xlabel("$k$")
plt.ylabel("$e_k$")
plt.show()
plt.plot(list(range(len(time_arr))), time_arr)
plt.xlabel("$k$")
plt.ylabel("Время")
plt.show()

print(np.mean(time_arr))
