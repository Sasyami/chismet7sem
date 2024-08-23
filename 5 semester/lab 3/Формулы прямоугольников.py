import numpy as np
import matplotlib.pyplot as plt


# интегрируемая функция
def F(x):
    return 1 / (2 * x ** 2 + 1)


# задание интервала
x_beginning = -1
x_ending = 1

# вычисленное значение интеграла
I = np.sqrt(2) * np.arctan(np.sqrt(2))


# формула левых прямоугольников

def left_rectangle_rule(x, h):
    left_rectangle_sum = 0
    for i in range(len(x)):
        left_rectangle_sum += F(x[i]) * h
    return left_rectangle_sum


# формула средних прямоугольников

def central_rectangle_rule(x, h):
    central_rectangle_sum = 0
    for i in range(len(x) - 1):
        central_rectangle_sum += F((x[i] + x[i+1]) / 2) * h
    return central_rectangle_sum


# формула правых прямоугольников

def right_rectangle_rule(x, h):
    right_rectangle_sum = 0
    for i in range(len(x) - 1):
        right_rectangle_sum += F(x[i+1]) * h
    return right_rectangle_sum


# задание числа узлов сетки
n = 10

h_ = []
left_rectangle_sum = []
central_rectangle_sum = []
right_rectangle_sum = []


for i in range(18):
    # узлы
    x = np.linspace(x_beginning, x_ending, n)

    # шаг
    h = (x_ending - x_beginning) / (n - 1)

    left_rectangle_sum.append(left_rectangle_rule(x, h))
    central_rectangle_sum.append(central_rectangle_rule(x, h))
    right_rectangle_sum.append(right_rectangle_rule(x, h))

    h_.append(h)

    n *= 2

I_ = [I for i in h_]

plt.suptitle("Численное интегрирование функции f(x)", fontsize=11)

# случай, когда используется формула левых прямоугольников
sp1 = plt.subplot(131)
plt.plot(h_, I_, color='black', label=r'''∫f(x)dx''')
plt.plot(h_, left_rectangle_sum, color='lightskyblue', label=r'''Формула левых прямоугольников''')
plt.xlabel('$h$', fontsize=10)
plt.ylabel(r'''$∫f(x)dx$''', fontsize=10)
plt.legend(loc='upper left')
plt.grid(True)

# случай, когда используется формула средних прямоугольников
sp2 = plt.subplot(132)
plt.plot(h_, I_, color='black', label=r'''∫f(x)dx''')
plt.plot(h_, central_rectangle_sum, color='lightskyblue', label=r'''Формула средних прямоугольников''')
plt.xlabel('$h$', fontsize=10)
plt.ylabel(r'''$∫f(x)dx$''', fontsize=10)
plt.legend(loc='upper left')
plt.grid(True)

# случай, когда используется формула правых прямоугольников
sp3 = plt.subplot(133)
plt.plot(h_, I_, color='black', label=r'''∫f(x)dx''')
plt.plot(h_, right_rectangle_sum, color='lightskyblue', label=r'''Формула правых прямоугольников''')
plt.xlabel('$h$', fontsize=10)
plt.ylabel(r'''$∫f(x)dx$''', fontsize=10)
plt.legend(loc='lower left')
plt.grid(True)

plt.show()