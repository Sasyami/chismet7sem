import numpy as np
from matplotlib import pyplot as plt

σ = 0.5
k = 1

α_L = 0
β_L = -1


def μ_L(t):
    return t


α_R = 2
β_R = 1


def μ_R(t):
    return 2 + 2 * np.sinh(t) + t * np.cosh(t)


def Ⴔ(x):
    return x ** 2 / 2


def f(x, t):
    return -1 + x * np.cosh(x * t) - t ** 2 * np.sinh(x * t)


def B_(h, Ꚍ, x):
    B = np.zeros(len(x))

    B[0] = β_L / (h * α_L + β_L)
    for j in range(1, len(x) - 1):
        B[j] = Ꚍ * σ / (Ꚍ * σ * (2 - B[j - 1]) + h ** 2)
    return B


def A_(h, Ꚍ, x, t, idx, u):
    A = np.zeros(len(x))
    B = B_(h, Ꚍ, x)

    A[0] = h * μ_L(t[int(idx)]) / (h * α_L + β_L)
    for j in range(1, len(x) - 1):
        A[j] = (Ꚍ * h ** 2 / (Ꚍ * σ * (2 - B[j - 1]) + h ** 2)) * (f(x[j], (t[int(idx) - 1] + t[int(idx)]) / 2) + (1 - σ) * (u[j + 1] - 2 * u[j] + u[j - 1]) / h ** 2 + u[j] / Ꚍ + A[j - 1] * σ / h ** 2)
    A[len(x) - 1] = (h * μ_R(t[int(idx)]) + β_R * A[len(x) - 2]) / (h * α_R + β_R * (1 - B[len(x) - 2]))
    return A


def Krank_Nicholson_scheme(h, Ꚍ, x, t, idx_t):
    u_current = np.zeros(len(x))
    u_tmp = np.zeros(len(x))

    for j in range(len(x)):
        u_current[j] = Ⴔ(x[j])

    for n in range(0, int(idx_t)):
        A = A_(h, Ꚍ, x, t, n + 1, u_current)
        B = B_(h, Ꚍ, x)

        u_tmp[len(x) - 1] = A[len(x) - 1]
        for j in range(len(x) - 2, -1, -1):
            u_tmp[j] = A[j] + B[j] * u_tmp[j + 1]

        for j in range(len(x)):
            u_current[j] = u_tmp[j]

    return u_current


def exact(x, t, idx_t):
    u = np.zeros(len(x))

    for j in range(len(x)):
        u[j] = x[j] ** 2 / 2 + np.sinh(x[j] * t[int(idx_t)])

    return u


fault_array = []


def max_err(h, Ꚍ, x, t, idx_t):
    u1 = Krank_Nicholson_scheme(h, Ꚍ, x, t, idx_t)
    u2 = exact(x, t, idx_t)

    err = [np.abs(u1[j] - u2[j]) for j in range(len(x))]
    fault_array.append(np.log(np.max(err)))


h = 0.01
ln_h = []

Ꚍ = 0.01
t = np.arange(0.0, 1.0, Ꚍ)

for i in range(3):
    x = np.arange(0.0, 1.0, h)
    ln_h.append(np.log(h))

    max_err(h, Ꚍ, x, t, len(t) - 1)

    h /= 3

tg = np.abs(fault_array[0] - fault_array[2]) / np.abs(ln_h[0] - ln_h[2])
print(tg)

plt.suptitle('Оценка порядка аппроксимации', fontsize=11)
plt.plot(ln_h, fault_array, color='black')
plt.xlabel('ln(h)', fontsize=10)
plt.ylabel('ln(max_err)', fontsize=10)
plt.grid(True)

plt.show()