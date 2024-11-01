import numpy as np
import matplotlib.pyplot as plt


# дифференцируемая функция
def F(x):
    return np.e ** (-x) * np.sin(x)


# вторая производная функции
def F_second_derivative(x):
    return -2 * np.e ** (-x) * np.cos(x)


# задание интервала
x_beginning = -0.8
x_ending = 0.8

# задание числа узлов сетки
n = 30

# узлы
x = np.linspace(x_beginning, x_ending, n)

# шаг
h = (x_ending - x_beginning) / (n - 1)

# значения второй производной функции в узлах сетки
f_second_derivative = [F_second_derivative(i) for i in x]


# 2 производная 2 порядка точности
_2_derivative_of_2_order = []


def second_order(x):
    for i in range(len(x)):
        if i == 0:
            _2_derivative_of_2_order.append((F(x[i + 1]) - 2 * F(x[i]) + F(x[i] - h)) / (h ** 2))  # выход за пределы интервала на h
        elif i == len(x) - 1:
            _2_derivative_of_2_order.append((F(x[i] + h) - 2 * F(x[i]) + F(x[i - 1])) / (h ** 2))  # выход за пределы интервала на h
        else:
            _2_derivative_of_2_order.append((F(x[i + 1]) - 2 * F(x[i]) + F(x[i - 1])) / (h ** 2))  # стандартная формула для центральной разности

    return _2_derivative_of_2_order


# 2 производная 4 порядка точности
_2_derivative_of_4_order = []


def forth_order(x):
    for i in range(len(x)):
        if i == 0:
            _2_derivative_of_4_order.append((-F(x[i + 2]) + 16 * F(x[i + 1]) - 30 * F(x[i]) + 16 * F(x[i] - h) - F(x[i] - 2 * h)) / (12 * h ** 2))  # выход за пределы интервала на 2h
        elif i == 1:
            _2_derivative_of_4_order.append((-F(x[i + 2]) + 16 * F(x[i + 1]) - 30 * F(x[i]) + 16 * F(x[i - 1]) - F(x[i - 1] - h)) / (12 * h ** 2))  # выход за пределы интервала на h
        elif i == len(x) - 1:
            _2_derivative_of_4_order.append((-F(x[i] + 2 * h) + 16 * F(x[i] + h) - 30 * F(x[i]) + 16 * F(x[i - 1]) - F(x[i - 2])) / (12 * h ** 2))  # выход за пределы интервала на 2h
        elif i == len(x) - 2:
            _2_derivative_of_4_order.append((-F(x[i + 1] + h) + 16 * F(x[i + 1]) - 30 * F(x[i]) + 16 * F(x[i - 1]) - F(x[i - 2])) / (12 * h ** 2))  # выход за пределы интервала на h
        else:
            _2_derivative_of_4_order.append((-F(x[i + 2]) + 16 * F(x[i + 1]) - 30 * F(x[i]) + 16 * F(x[i - 1]) - F(x[i - 2])) / (12 * h ** 2))  # стандартная формула для центральной разности

    return _2_derivative_of_4_order


plt.suptitle("Численное дифференцирование функции f(x)", fontsize=11)

# случай, когда вычисляется вторая производная 2 порядка точности
sp1 = plt.subplot(121)
plt.plot(x, f_second_derivative, color='black', label=r'''f''(x)''')
plt.plot(x, second_order(x), color='lightskyblue', label=r'''2 порядок, n = 30''')
plt.xlabel('$x$', fontsize=10)
plt.ylabel(r'''$f''(x)$''', fontsize=10)
plt.legend(loc='upper left')
plt.grid(True)

# случай, когда вычисляется вторая производная 4 порядка точности
sp2 = plt.subplot(122)
plt.plot(x, f_second_derivative, color='black', label=r'''f''(x)''')
plt.plot(x, forth_order(x), color='violet', label=r'''4 порядок, n = 30''')
plt.xlabel('$x$', fontsize=10)
plt.ylabel(r'''$f''(x)$''', fontsize=10)
plt.legend(loc='upper left')
plt.grid(True)

plt.show()
