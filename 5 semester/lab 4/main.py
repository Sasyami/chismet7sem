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


print("Метод дихотомии")
print("Значение корня уравнения с точностью ε = 10^(-3): ", the_dichotomy_method(10**(-3)))
print("Значение корня уравнения с точностью ε = 10^(-6): ", the_dichotomy_method(10**(-6)))
print("Значение корня уравнения с точностью ε = 10^(-9): ", the_dichotomy_method(10**(-9)))
print(" ")
print("Метод Ньютона")
print("Значение корня уравнения с точностью ε = 10^(-3): ", handling_the_divergence_case(10**(-3)))
print("Значение корня уравнения с точностью ε = 10^(-6): ", handling_the_divergence_case(10**(-6)))
print("Значение корня уравнения с точностью ε = 10^(-9): ", handling_the_divergence_case(10**(-9)))
