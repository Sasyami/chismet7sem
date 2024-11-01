import numpy as np
import matplotlib.pyplot as plt

# преобразуем исходное уравнение в систему


def V(x, u, y):
    return y


def U(x, u, y):
    return (-y + (1 + 2 * x) * u) / (2 * (1 + x)) + (3 * np.cos(x) - (3 + 4 * x) * np.sin(x)) / (2 * np.sqrt(1 + x))


# задание интервала
x_beginning = 0
x_ending = 1


# метод Рунге-Кутты (4-ый порядок точности)


def Runge_Kutta(h):
    # число узлов сетки
    n = 1 + (x_ending - x_beginning) / h

    # узлы
    x = np.linspace(x_beginning, x_ending, int(n))

    u = [1]
    y = [0]

    for i in range(len(x) - 1):
        q0 = U(x[i], u[i], y[i])
        k0 = V(x[i], u[i], y[i])
        q1 = U(x[i] + h / 2, u[i] + k0 * h / 2, y[i] + q0 * h / 2)
        k1 = V(x[i] + h / 2, u[i] + k0 * h / 2, y[i] + q0 * h / 2)
        q2 = U(x[i] + h / 2, u[i] + k1 * h / 2, y[i] + q1 * h / 2)
        k2 = V(x[i] + h / 2, u[i] + k1 * h / 2, y[i] + q1 * h / 2)
        q3 = U(x[i] + h, u[i] + k2 * h, y[i] + q2 * h)
        k3 = V(x[i] + h, u[i] + k2 * h, y[i] + q2 * h)

        u.append(u[i] + (h / 6) * (k0 + 2 * k1 + 2 * k2 + k3))
        y.append(y[i] + (h / 6) * (q0 + 2 * q1 + 2 * q2 + q3))

    return u


u = Runge_Kutta(0.05)
u_ = Runge_Kutta(0.1)

Runge_rule = [(1 / 15) * abs(u[j] - u_[j]) for j in range(len(u_))]
print(Runge_rule)

# число узлов сетки
n = 1 + (x_ending - x_beginning) / 0.1

# узлы
x = np.linspace(x_beginning, x_ending, int(n))

plt.plot(x, Runge_rule, color='black')
plt.xlabel('x', fontsize=10)
plt.ylabel('Runge_rule', fontsize=10)
plt.grid(True)

plt.show()