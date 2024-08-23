from wrapt_timeout_decorator import timeout
import numpy as np
import matplotlib.pyplot as plt

# заданная функция
def F(x):
    return np.log(x ** 2 + 3 * x + 1) - np.cos(2 * x + 1)


# производная функции
def F_derivative(x):
    return (2 * x + 3) / (x ** 2 + 3 * x + 1) + 2 * np.sin(2 * x + 1)


# задание интервала
x_beginning = 0
x_ending = 10

# узлы интерполяции
_x_ = np.linspace(x_beginning, x_ending, 100)

# метод дихотомии
def the_dichotomy_method(ε):
    a = x_beginning
    b = x_ending

    for i in range(round(np.log2((x_ending - x_beginning) / ε)) + 1):
        x = (a + b) / 2
        if F(x) * F(a) == 0:
            return x
        elif F(x) * F(a) < 0:
            b = x
        else:
            a = x

    return x


# метод Ньютона

# устанавливаем таймер на 3 секунды, чтобы обработать переполнение при расходимости метода
@timeout(3)
def newton_s_method(b, ε):

    # обработка деления на ноль
    try:
        x = b - F(b) / F_derivative(b)
    except ZeroDivisionError:
        return 1

    # если значение вышло за пределы области поиска, возвращаем его, пользуясь идеей половинного деления
    while x >= x_ending or x <= x_beginning:
        x = (b + x) / 2

    while np.abs(x - b) > ε:
        b = x

        # обработка деления на ноль
        try:
            x = b - F(b) / F_derivative(b)
        except ZeroDivisionError:
            return 1

        # обработка случая выхода за пределы интервала
        while x >= x_ending or x <= x_beginning:
            x = (b + x) / 2

    # программа сработала без ошибок
    return 0


def handling_TimeoutError(b, ε):
    try:
        return newton_s_method(b, ε)
    except TimeoutError:
        return 2


# метод Ньютона может расходиться при неправильном выборе начального приближения,
# эта функция позволяет найти верную точку старта (b) и получить приближенное значение корня
def handling_the_divergence_case(ε):

    # поиск точки старта
    b = x_ending
    while not(handling_TimeoutError(b, ε) == 0):
        b = (b + x_beginning) / 2

    x = b - F(b) / F_derivative(b)
    while x >= x_ending or x <= x_beginning:
        x = (b + x) / 2

    while np.abs(x - b) > ε:
        b = x
        x = b - F(b) / F_derivative(b)
        while x >= x_ending or x <= x_beginning:
            x = (b + x) / 2

    return x


x_ = []
f_ = []


def tangents():
    b = x_ending
    x_.append(b)
    f_.append(F(b))
    i = 0

    x = b - F(b) / F_derivative(b)
    x_.append(x)
    f_.append(F(x))

    while i != 5:
        b = x
        x_.append(b)
        f_.append(F(b))
        x = b - F(b) / F_derivative(b)
        x_.append(x)
        f_.append(F(x))
        i += 1


tangents()
plt.suptitle("Метод Ньютона для функции f(x)", fontsize=11)


plt.plot(_x_, F(_x_), color='black')
plt.plot([x_[0], x_[1]], [f_[0], 0], color='lightskyblue')
plt.plot([x_[2], x_[3]], [f_[2], 0], color='lightskyblue')
plt.plot([x_[4], x_[5]], [f_[4], 0], color='lightskyblue')
plt.grid(True)
plt.show()