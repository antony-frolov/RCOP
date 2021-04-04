import matplotlib.pyplot as plt
import numpy as np

plt.scatter([15 + 20, 20 + 30, 30 + 45], [1.53, 7.808, 89.52])
plt.plot(np.linspace(0, 100, 200), [n ** 6 / 2001152663 for n in np.linspace(0, 100, 200)])
plt.xlabel("$m + n$")
plt.ylabel("Время")
plt.show()
print(np.array([n ** 6 / 2001152663 for n in [15 + 20, 20 + 30, 30 + 45]]) - [1.53, 7.808, 89.52])


plt.scatter([20 * 10, 30 * 15, 45 * 20], [1.95, 10.504, 49.43])
plt.plot(np.linspace(0, 2500, 200), [n ** 2 / 19278.37 for n in np.linspace(0, 2500, 200)])
plt.xlabel("$nr$")
plt.ylabel("Время")
plt.show()
print(np.array([n ** 2 / 19278.37 for n in [20 * 10, 30 * 15, 45 * 20]]) - [1.95, 10.504, 49.43])
