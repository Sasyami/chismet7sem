import numpy as np
import matplotlib.pyplot as plt

# МЕТОД ПРОГОНКИ с краевыми условиями второго порядка точности

# точное решение


def exact_solution(x):
    return np.sqrt(1 + x) * np.sin(x) + np.exp(-x)


def p(x):
    return 1 / (2 * (1 + x))


def q(x):
    return -(1 + 2 * x) / (2 * (1 + x))


def f(x):
    return (3 * np.cos(x) - (3 + 4 * x) * np.sin(x)) / (2 * np.sqrt(1 + x))


alpha1 = 0
alpha2 = -2
beta1 = 1
beta2 = 1
gamma1 = 1
gamma2 = 0.1704


def a_k(x, h):
    return 1 / h**2 - p(x) / (2 * h)


def b_k(x, h):
    return -2 / h**2 + q(x)


def c_k(x, h):
    return 1 / h**2 + p(x) / (2 * h)


b_0 = beta1
c_0 = 0
f_0 = gamma1


def a_n(h):
    return 2 / h**2


def b_n(h):
    return -2 / h**2 + q(x_ending) - 2 * beta2 / (h * alpha2) - p(x_ending) * beta2 / alpha2


def f_n(h):
    return f(x_ending) - p(x_ending) * gamma2 / alpha2 - 2 * gamma2 / (h * alpha2)


def A_k(x, h):
    A_k_ = [-c_0 / b_0]
    for i in range(1, len(x) - 1):
        A_k_.append(-c_k(x[i], h) / (b_k(x[i], h) + a_k(x[i], h) * A_k_[i - 1]))
    A_k_.append(0)
    return A_k_


def B_k(x, h):
    A_k_ = A_k(x, h)
    B_k_ = [f_0 / b_0]
    for i in range(1, len(x) - 1):
        B_k_.append((f(x[i]) - a_k(x[i], h) * B_k_[i - 1]) / (b_k(x[i], h) + a_k(x[i], h) * A_k_[i - 1]))
    B_k_.append((f_n(h) - a_n(h) * B_k_[len(x) - 2]) / (b_n(h) + a_n(h) * A_k_[len(x) - 2]))
    return B_k_


fault_array = []


def run_through_method(x, h):
    A_k_ = A_k(x, h)
    B_k_ = B_k(x, h)
    u = [B_k_[len(x) - 1]]
    for i in range(len(x) - 2, -1, -1):
        u.insert(0, B_k_[i] + A_k_[i] * u[0])

    err = [abs(exact_solution(x[j]) - u[j]) for j in range(len(x))]
    fault_array.append(np.log(max(err)))


# задание интервала
x_beginning = 0
x_ending = 1

# шаг
h = 0.05

ln_h = []

for i in range(5):
    # число узлов сетки
    n = 1 + (x_ending - x_beginning) / h

    # узлы
    x = np.linspace(x_beginning, x_ending, int(n))

    run_through_method(x, h)

    ln_h.append(np.log(h))

    h /= 2


tg = abs(fault_array[0] - fault_array[1]) / abs(ln_h[0] - ln_h[1])
print(tg)


plt.suptitle("Оценка точности метода прогонки", fontsize=11)

plt.plot(ln_h, fault_array, color='black')
plt.xlabel('ln(h)', fontsize=10)
plt.ylabel('ln(max_err)', fontsize=10)
plt.grid(True)

plt.show()