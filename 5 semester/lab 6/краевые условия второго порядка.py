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

# шаг
h = 0.05

# задание интервала
x_beginning = 0
x_ending = 1

# число узлов сетки
n = 1 + (x_ending - x_beginning) / h

# узлы
x = np.linspace(x_beginning, x_ending, int(n))


def a_k(x):
    return 1 / h**2 - p(x) / (2 * h)


def b_k(x):
    return -2 / h**2 + q(x)


def c_k(x):
    return 1 / h**2 + p(x) / (2 * h)


b_0 = beta1
c_0 = 0
f_0 = gamma1

a_n = 2 / h**2
b_n = -2 / h**2 + q(x_ending) - 2 * beta2 / (h * alpha2) - p(x_ending) * beta2 / alpha2
f_n = f(x_ending) - p(x_ending) * gamma2 / alpha2 - 2 * gamma2 / (h * alpha2)

A_k_ = [-c_0 / b_0]


def A_k(x):
    for i in range(1, len(x) - 1):
        A_k_.append(-c_k(x[i]) / (b_k(x[i]) + a_k(x[i]) * A_k_[i - 1]))
    A_k_.append(0)


A_k(x)


B_k_ = [f_0 / b_0]


def B_k(x):
    for i in range(1, len(x) - 1):
        B_k_.append((f(x[i]) - a_k(x[i]) * B_k_[i - 1]) / (b_k(x[i]) + a_k(x[i]) * A_k_[i - 1]))
    B_k_.append((f_n - a_n * B_k_[len(x) - 2]) / (b_n + a_n * A_k_[len(x) - 2]))


B_k(x)


u = [B_k_[len(x)-1]]


def run_through_method(x):
    for i in range(len(x) - 2, -1, -1):
        u.insert(0, B_k_[i] + A_k_[i] * u[0])


run_through_method(x)


plt.suptitle("Приближенное решение краевой задачи для ОДУ", fontsize=11)

plt.plot(x, u, color='lightskyblue', label='numerical solution')
plt.plot(x, exact_solution(x), color='black', label='exact solution')
plt.xlabel('x', fontsize=10)
plt.ylabel('u(x)', fontsize=10)
plt.legend(loc='upper left')
plt.grid(True)

plt.show()