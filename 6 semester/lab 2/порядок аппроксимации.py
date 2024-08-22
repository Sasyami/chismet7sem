import numpy as np

h1 = 0.01
h2 = 0.001
c = 1
a = 0.5

Ꚍ1 = c * h1 / a
Ꚍ2 = c * h2 / a

x0 = 0.5
દ = 0.2


def ξ(x):
    return np.abs(x - x0) / દ


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


x1 = np.arange(0.0, 5.0, h1)
t1 = np.arange(0.0, 5.0, Ꚍ1)

x2 = np.arange(0.0, 5.0, h2)
t2 = np.arange(0.0, 5.0, Ꚍ2)


def difference_scheme(a, h, Ꚍ, x, t, Ⴔ, 𐓑, μ, idx_t):
    u_current = np.zeros(len(x))
    u_tmp = np.zeros(len(x))

    for j in range(len(x)):
        u_current[j] = Ⴔ(x[j])

    for n in range(0, int(idx_t)):
        for j in range(1, len(x) - 1):
            u_tmp[j] = u_current[j] + Ꚍ * (𐓑(x[j], t[n]) - a * ((u_current[j + 1] - u_current[j - 1]) / (2 * h)) + (
                        Ꚍ * a ** 2 / (2 * h ** 2)) * (u_current[j + 1] - 2 * u_current[j] + u_current[j - 1]))
        u_tmp[0] = μ(t[n + 1])
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


idx_t = len(t1) - 1

num_solution_1 = difference_scheme(a, h1, Ꚍ1, x1, t1, Ⴔ1, 𐓑, Ⴔ1, idx_t)
exact_solution_1 = exact(a, x1, t1, Ⴔ1, Ⴔ1, idx_t)

num_solution_2 = difference_scheme(a, h2, Ꚍ2, x2, t2, Ⴔ1, 𐓑, Ⴔ1, idx_t * 10)
exact_solution_2 = exact(a, x2, t2, Ⴔ1, Ⴔ1, idx_t * 10)

દ1 = np.max(np.abs(num_solution_1 - exact_solution_1))
દ2 = np.max(np.abs(num_solution_2 - exact_solution_2))

print(દ1 / દ2)