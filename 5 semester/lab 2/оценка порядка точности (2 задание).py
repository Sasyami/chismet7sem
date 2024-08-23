import numpy as np
import matplotlib.pyplot as plt


# дифференцируемая функция
def F(x):
    return np.e ** (-x) * np.sin(x)


# вторая производная функции
def F_second_derivative(x):
    return -2 * np.e ** (-x) * np.cos(x)


_2_fault_array = []


def ln_max_err_2(x, h):
    # 2 производная 2 порядка точности
    _2_derivative_of_2_order = []

    for i in range(len(x)):
        if i == 0:
            _2_derivative_of_2_order.append((F(x[i + 1]) - 2 * F(x[i]) + F(x[i] - h)) / (h ** 2))  # выход за пределы интервала на h
        elif i == len(x) - 1:
            _2_derivative_of_2_order.append((F(x[i] + h) - 2 * F(x[i]) + F(x[i - 1])) / (h ** 2))  # выход за пределы интервала на h
        else:
            _2_derivative_of_2_order.append((F(x[i + 1]) - 2 * F(x[i]) + F(x[i - 1])) / (h ** 2))  # стандартная формула для центральной разности

    err = [abs(F_second_derivative(x[j]) - _2_derivative_of_2_order[j]) for j in range(len(x))]
    _2_fault_array.append(np.log(max(err)))


_4_fault_array = []


def ln_max_err_4(x, h):
    # 2 производная 4 порядка точности
    _2_derivative_of_4_order = []

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

    err = [abs(F_second_derivative(x[j]) - _2_derivative_of_4_order[j]) for j in range(len(x))]
    _4_fault_array.append(np.log(max(err)))


# задание интервала
x_beginning = -0.8
x_ending = 0.8

# задание числа узлов сетки
n = 10

ln_h = []

for i in range(6):
    # узлы
    x = np.linspace(x_beginning, x_ending, n)

    # шаг
    h = (x_ending - x_beginning) / (n - 1)

    ln_max_err_2(x, h)
    ln_max_err_4(x, h)

    ln_h.append(np.log(h))

    n *= 2


plt.suptitle("Оценка порядка точности метода конечных разностей", fontsize=11)

# случай, когда вычисляется вторая производная 2 порядка точности
sp1 = plt.subplot(121)
plt.plot(ln_h, _2_fault_array, color='black')
plt.xlabel('ln(h)', fontsize=10)
plt.ylabel('ln(max_err)', fontsize=10)
plt.grid(True)

# случай, когда вычисляется вторая производная 4 порядка точности
sp2 = plt.subplot(122)
plt.plot(ln_h, _4_fault_array, color='black')
plt.xlabel('ln(h)', fontsize=10)
plt.ylabel('ln(max_err)', fontsize=10)
plt.grid(True)

plt.show()