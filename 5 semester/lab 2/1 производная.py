import numpy as np
import matplotlib.pyplot as plt


# дифференцируемая функция
def F(x):
    return np.e ** (-x) * np.sin(x)


# производная функции
def F_derivative(x):
    return np.e ** (-x) * (np.cos(x) - np.sin(x))


# задание интервала
x_beginning = -0.8
x_ending = 0.8

# задание числа узлов сетки
n = 30

# узлы
x = np.linspace(x_beginning, x_ending, n)

# шаг
h = (x_ending - x_beginning) / (n - 1)

# значения производной функции в узлах сетки
f_derivative = [F_derivative(i) for i in x]

# правая разность
forward_dif = []


def forward_difference(x):
    for i in range(len(x)):
        if i == len(x) - 1:
            forward_dif.append((F(x[i]) - F(x[i - 1])) / h)  # для крайней правой точки считаем значение производной через левую разность, поскольку порядок точности одинаков
        else:
            forward_dif.append((F(x[i + 1]) - F(x[i])) / h)  # стандартная формула для правой разности

    return forward_dif


# центральная разность
central_dif = []


def central_difference(x):
    for i in range(len(x)):
        if i == 0:
            central_dif.append((-3 * F(x[i]) + 4 * F(x[i + 1]) - F(x[i + 2])) / (2 * h))  # для крайней левой точки
        elif i == len(x) - 1:
            central_dif.append((3 * F(x[i]) - 4 * F(x[i - 1]) + F(x[i - 2])) / (2 * h))  # для крайней правой точки
        else:
            central_dif.append((F(x[i + 1]) - F(x[i - 1])) / (2 * h))  # стандартная формула для центральной разности

    return central_dif


plt.suptitle("Численное дифференцирование функции f(x)", fontsize=11)

# случай, когда используется правая разность
sp1 = plt.subplot(121)
plt.plot(x, f_derivative, color='black', label=r'''f'(x)''')
plt.plot(x, forward_difference(x), color='lightskyblue', label=r'''Правая разность, n = 30''')
plt.xlabel('$x$', fontsize=10)
plt.ylabel(r'''$f'(x)$''', fontsize=10)
plt.legend(loc='upper left')
plt.grid(True)

# случай, когда используется центральная разность
sp2 = plt.subplot(122)
plt.plot(x, f_derivative, color='black', label=r'''f'(x)''')
plt.plot(x, central_difference(x), color='violet', label=r'''Центральная разность, n = 30''')
plt.xlabel('$x$', fontsize=10)
plt.ylabel(r'''$f'(x)$''', fontsize=10)
plt.legend(loc='upper left')
plt.grid(True)

plt.show()