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


# задание интервала
x_beginning = 0
x_ending = 1

# шаг
h = 0.05

# число узлов сетки
n = 1 + (x_ending - x_beginning) / h

# узлы
x = np.linspace(x_beginning, x_ending, int(n))


# значения функции exact_solution в узлах сетки
f_exact = [exact_solution(i) for i in x]


# явный метод Эйлера (1-ый порядок точности)


def Euler(x):
    u = [1]
    y = [0]

    for i in range(len(x) - 1):
        q0 = U(x[i], u[i], y[i])
        k0 = V(x[i], u[i], y[i])
        u.append(u[i] + h * k0)
        y.append(y[i] + h * q0)

    return u


# метод Рунге-Кутты (4-ый порядок точности)


def Runge_Kutta(x):
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


# метод Адамса (3-ий порядок точности)


def Adams(x):
    u = [1]
    y = [0]

    # первое и второе значения считаем по методу РК 3-го порядка
    for i in range(2):
        q0 = U(x[i], u[i], y[i]) * h
        k0 = V(x[i], u[i], y[i]) * h
        q1 = U(x[i] + h / 2, u[i] + k0 / 2, y[i] + q0 / 2) * h
        k1 = V(x[i] + h / 2, u[i] + k0 / 2, y[i] + q0 / 2) * h
        q2 = U(x[i] + h, u[i] - k0 + 2 * k1, y[i] - q0 + 2 * q1) * h
        k2 = V(x[i] + h, u[i] - k0 + 2 * k1, y[i] - q0 + 2 * q1) * h

        u.append(u[i] + (1 / 6) * (k0 + 4 * k1 + k2))
        y.append(y[i] + (1 / 6) * (q0 + 4 * q1 + q2))

    # метод Адамса 3-го порядка
    for i in range(2, len(x) - 1):
        q0 = U(x[i], u[i], y[i]) * h
        k0 = V(x[i], u[i], y[i]) * h
        q1 = U(x[i-1], u[i-1], y[i-1]) * h
        k1 = V(x[i-1], u[i-1], y[i-1]) * h
        q2 = U(x[i-2], u[i-2], y[i-2]) * h
        k2 = V(x[i-2], u[i-2], y[i-2]) * h

        u.append(u[i] + (1 / 12) * (23 * k0 - 16 * k1 + 5 * k2))
        y.append(y[i] + (1 / 12) * (23 * q0 - 16 * q1 + 5 * q2))

    return u


plt.suptitle("Приближенное решение задачи Коши", fontsize=11)

# метод Эйлера
sp1 = plt.subplot(131)
plt.plot(x, Euler(x), color='lightskyblue', label='numerical solution (Euler)')
plt.plot(x, exact_solution(x), color='black', label='exact solution')
plt.xlabel('x', fontsize=10)
plt.ylabel('u(x)', fontsize=10)
plt.legend(loc='upper left')
plt.grid(True)

# метод Рунге-Кутты
sp2 = plt.subplot(132)
plt.plot(x, Runge_Kutta(x), color='lightskyblue', label='numerical solution (RK)')
plt.plot(x, exact_solution(x), color='black', label='exact solution')
plt.xlabel('x', fontsize=10)
plt.ylabel('u(x)', fontsize=10)
plt.legend(loc='upper left')
plt.grid(True)

# метод Адамса
sp3 = plt.subplot(133)
plt.plot(x, Adams(x), color='lightskyblue', label='numerical solution (Adams)')
plt.plot(x, exact_solution(x), color='black', label='exact solution')
plt.xlabel('x', fontsize=10)
plt.ylabel('u(x)', fontsize=10)
plt.legend(loc='upper left')
plt.grid(True)

plt.show()