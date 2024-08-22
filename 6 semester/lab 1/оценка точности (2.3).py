import numpy as np
from matplotlib import pyplot as plt

h = 0.01
c = 0.5
a = 0.5

Ꚍ = c * h / a

x0 = 0.5
w = 10


def g(t):
    return np.arccos(np.cos(w * t)) ** 4 * 0.01


def 𐓑(x, x0, t):
    if x == x0: return g(t) / h
    else: return 0


def difference_scheme(a, h, Ꚍ, x, t, 𐓑, idx_t):
    u_current = np.zeros(len(x))
    u_tmp = np.zeros(len(x))

    for j in range(len(x)):
        u_current[j] = 0

    for n in range(0, int(idx_t)):
        for j in range(1, len(x)):
            u_tmp[j] = u_current[j] + Ꚍ * (𐓑(x[j], x0, t[n]) - a * (u_current[j] - u_current[j - 1]) / h)
        u_tmp[0] = 0

        for j in range(len(x)):
            u_current[j] = u_tmp[j]

    return u_current


def exact(a, x, t, idx_t):
    u = np.zeros(len(x))

    for j in range(len(x)):
        if 0 < x[j] - x0 < a * t[int(idx_t)]:
            u[j] = (1 / a) * g(t[int(idx_t)] + (x0 - x[j]) / a)
        else:
            u[j] = 0

    return u


fault_array = []


def max_err(a, h, Ꚍ, x, t, 𐓑, idx_t):
    u1 = difference_scheme(a, h, Ꚍ, x, t, 𐓑, idx_t)
    u2 = exact(a, x, t, idx_t)

    err = [np.abs(u1[j] - u2[j]) for j in range(len(x))]
    fault_array.append(np.log(np.max(err)))


h = 0.01
ln_h = []

for i in range(3):
    x = np.arange(0.0, 5.0, h)
    ln_h.append(np.log(h))

    Ꚍ = c * h / a
    t = np.arange(0.0, 5.0, Ꚍ)

    max_err(a, h, Ꚍ, x, t, 𐓑, len(t) - 1)

    h /= 6

tg = np.abs(fault_array[1] - fault_array[2]) / np.abs(ln_h[1] - ln_h[2])
print(tg)

plt.suptitle('Оценка порядка аппроксимации', fontsize=11)
plt.plot(ln_h, fault_array, color='black')
plt.xlabel('ln(h)', fontsize=10)
plt.ylabel('ln(max_err)', fontsize=10)
plt.grid(True)

plt.show()