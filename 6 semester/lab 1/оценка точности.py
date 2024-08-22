import numpy as np
from matplotlib import pyplot as plt

a = 0.5
c = 0.5

x0 = 0.5
દ = 0.2


def ξ(x):
    return np.abs(x-x0) / દ


def Ⴔ1(x):
    return np.heaviside(1 - ξ(x), 1)


def Ⴔ2(x):
    return Ⴔ1(x) * (1 - ξ(x) ** 2)


def Ⴔ3(x):
    return Ⴔ1(x) * np.exp(- ξ(x) ** 2 / (np.abs(1 - ξ(x) ** 2)))


def Ⴔ4(x):
    return Ⴔ1(x) * (np.cos(np.pi * ξ(x) / 2) ** 3)


def 𐓑(x, t):
    return 0


def difference_scheme(a, h, Ꚍ, x, t, Ⴔ, 𐓑, μ, idx_t):
    u_current = np.zeros(len(x))
    u_tmp = np.zeros(len(x))

    for j in range(len(x)):
        u_current[j] = Ⴔ(x[j])

    for n in range(0, int(idx_t)):
        for j in range(1, len(x)):
            u_tmp[j] = u_current[j] + Ꚍ * (𐓑(x[j], t[n]) - a * (u_current[j] - u_current[j - 1]) / h)
        u_tmp[0] = μ(t[n + 1])

        for j in range(len(x)):
            u_current[j] = u_tmp[j]

    return u_current


def exact(a, x, t, Ⴔ, μ, idx_t):
    u = np.zeros(len(x))

    for j in range(len(x)):
        if x[j] >= a * t[int(idx_t)]:
            u[j] = Ⴔ(x[j] - a * t[int(idx_t)])
        else:
            u[j] = μ(t[int(idx_t)] - x[j] / a)

    return u


fault_array = []


def max_err(a, h, Ꚍ, x, t, Ⴔ, 𐓑, μ, idx_t):
    u1 = difference_scheme(a, h, Ꚍ, x, t, Ⴔ, 𐓑, μ, idx_t)
    u2 = exact(a, x, t, Ⴔ, μ, idx_t)

    err = [np.abs(u1[j] - u2[j]) for j in range(len(x))]
    fault_array.append(np.log(np.max(err)))


h = 0.01
ln_h = []

for i in range(3):
    x = np.arange(0.0, 5.0, h)
    ln_h.append(np.log(h))

    Ꚍ = c * h / a
    t = np.arange(0.0, 5.0, Ꚍ)

    max_err(a, h, Ꚍ, x, t, Ⴔ4, 𐓑, Ⴔ4, len(t) - 1)

    h /= 5

tg = np.abs(fault_array[0] - fault_array[2]) / np.abs(ln_h[0] - ln_h[2])
print(tg)

plt.suptitle('Оценка порядка аппроксимации на $\\phi_4(x)$', fontsize=11)
plt.plot(ln_h, fault_array, color='black')
plt.xlabel('ln(h)', fontsize=10)
plt.ylabel('ln(max_err)', fontsize=10)
plt.grid(True)

plt.show()