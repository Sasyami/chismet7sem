import numpy as np
from matplotlib import pyplot as plt

a = 0.5
c = 0.5

x0 = 0.5
દ = 0.2
દ0 = 10 ** (-10)


def ξ(x):
    return np.abs(x-x0) / દ


def Ⴔ1(x):
    return np.heaviside(1 - ξ(x), 1 / 2)


def Ⴔ2(x):
    return Ⴔ1(x) * (1 - ξ(x) ** 2)


def Ⴔ3(x):
    return Ⴔ1(x) * np.exp(- ξ(x) ** 2 / (np.abs(1 - ξ(x) ** 2)))


def Ⴔ4(x):
    return Ⴔ1(x) * (np.cos(np.pi * ξ(x) / 2) ** 3)


def 𐓑(x, t):
    return 0


def flux_limiter(r):
    return 2 * r / (r ** 2 + 1)


def Lax_Wendroff_with_flow_correction(a, h, Ꚍ, x, t, Ⴔ, 𐓑, μ, idx_t):
    u_current = np.zeros(len(x))
    u_tmp = np.zeros(len(x))

    for j in range(len(x)):
        u_current[j] = Ⴔ(x[j])

    for n in range(0, int(idx_t)):
        for j in range(2, len(x) - 1):
            F_plus = a * u_current[j] + a * flux_limiter(
                (u_current[j] - u_current[j - 1]) / (u_current[j + 1] - u_current[j] + દ0)) * (((1 / 2) * (
                        u_current[j] + u_current[j + 1]) - ((a * Ꚍ) / (2 * h)) * (u_current[j + 1] - u_current[j])) -
                                                                                               u_current[j])
            F_minus = a * u_current[j - 1] + a * flux_limiter(
                (u_current[j - 1] - u_current[j - 2]) / (u_current[j] - u_current[j - 1] + દ0)) * (((1 / 2) * (
                        u_current[j - 1] + u_current[j]) - ((a * Ꚍ) / (2 * h)) * (u_current[j] - u_current[j - 1])) -
                                                                                                   u_current[j - 1])
            u_tmp[j] = u_current[j] + Ꚍ * (𐓑(x[j], t[n])) - (Ꚍ / h) * (F_plus - F_minus)
        u_tmp[0] = μ(t[n + 1])
        u_tmp[1] = u_current[1] + Ꚍ * (𐓑(x[1], t[n]) - a * (u_current[1] - u_current[0]) / h)
        u_tmp[len(x) - 1] = u_current[len(x) - 1] + Ꚍ * (𐓑(x[len(x) - 1], t[n]) - a * (u_current[len(x) - 1] - u_current[len(x) - 2]) / h)

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
    u1 = Lax_Wendroff_with_flow_correction(a, h, Ꚍ, x, t, Ⴔ, 𐓑, μ, idx_t)
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

    h /= 3

tg = np.abs(fault_array[0] - fault_array[2]) / np.abs(ln_h[0] - ln_h[2])
print(tg)

plt.suptitle('Оценка порядка аппроксимации на $\\phi_4(x)$', fontsize=11)
plt.plot(ln_h, fault_array, color='black')
plt.xlabel('ln(h)', fontsize=10)
plt.ylabel('ln(max_err)', fontsize=10)
plt.grid(True)

plt.show()