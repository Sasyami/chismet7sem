import numpy as np
import matplotlib.pyplot as plt

# преобразуем исходное уравнение в систему


def V(x, u, y):
    return y


def U(x, u, y):
    return (-y + (1 + 2 * x) * u) / (2 * (1 + x)) + (3 * np.cos(x) - (3 + 4 * x) * np.sin(x)) / (2 * np.sqrt(1 + x))


# точное решение


def exact_solution(x):
    return np.sqrt(1 + x) * np.sin(x) + np.exp(-x)


# метод Рунге-Кутты (4-ый порядок точности)
fault_array = []


def ln_max_err_Runge_Kutta(x, h):
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

    err = [abs(exact_solution(x[j]) - u[j]) for j in range(len(x))]
    fault_array.append(np.log(max(err)))


# задание интервала
x_beginning = 0
x_ending = 1

# шаг
h = 0.1
ln_h = []

for i in range(3):
    # число узлов сетки
    n = 1 + (x_ending - x_beginning) / h

    # узлы
    x = np.linspace(x_beginning, x_ending, int(n))

    ln_max_err_Runge_Kutta(x, h)
    ln_h.append(np.log(h))

    h /= 10


print(abs(fault_array[0] - fault_array[1]) / abs(ln_h[0] - ln_h[1]))


plt.suptitle("Оценка порядка точности метода РК4", fontsize=11)

plt.plot(ln_h, fault_array, color='black')
plt.xlabel('ln(h)', fontsize=10)
plt.ylabel('ln(max_err)', fontsize=10)
plt.grid(True)

plt.show()