import numpy as np
import matplotlib.pyplot as plt


# интегрируемая функция
def F(x):
    # return 1 / (2 * x ** 2 + 1)
    return x ** 3 + 2 * x - 5


# задание интервала
x_beginning = -1
x_ending = 1

# вычисленное значение интеграла
# I = np.sqrt(2) * np.arctan(np.sqrt(2))
I = -10


# формула средних прямоугольников

def rectangle_rule(x, h):
    rectangle_sum = 0
    for i in range(len(x) - 1):
        rectangle_sum += F((x[i] + x[i+1]) / 2) * h
    return rectangle_sum


# формула трапеции

def trapezoid_formula(x, h):
    trapezoid_sum = 0
    for i in range(len(x) - 1):
        trapezoid_sum += ((F(x[i]) + F(x[i+1])) / 2) * h
    return trapezoid_sum


# формула Симпсона

def Simpson_s_formula(x, h):
    Simpson_s_sum = 0
    for i in range(len(x) - 1):
        Simpson_s_sum += (h / 6) * (F(x[i]) + 4 * F((x[i] + x[i+1]) / 2) + F(x[i+1]))
    return Simpson_s_sum


# задание числа узлов сетки
n = 10

h_ = []
rectangle_sum = []
trapezoid_sum = []
Simpson_s_sum = []


for i in range(18):
    # узлы
    x = np.linspace(x_beginning, x_ending, n)

    # шаг
    h = (x_ending - x_beginning) / (n - 1)

    rectangle_sum.append(rectangle_rule(x, h))
    trapezoid_sum.append(trapezoid_formula(x, h))
    Simpson_s_sum.append(Simpson_s_formula(x, h))

    h_.append(h)

    n *= 2

I_ = [I for i in h_]

plt.suptitle("Численное интегрирование функции f(x)", fontsize=11)

# случай, когда используется формула средних прямоугольников
sp1 = plt.subplot(131)
plt.plot(h_, I_, color='black', label=r'''∫f(x)dx''')
plt.plot(h_, rectangle_sum, color='lightskyblue', label=r'''Формула средних прямоугольников''')
plt.xlabel('$h$', fontsize=10)
plt.ylabel(r'''$∫f(x)dx$''', fontsize=10)
plt.legend(loc='upper left')
plt.grid(True)

# случай, когда используется формула трапеции
sp2 = plt.subplot(132)
plt.plot(h_, I_, color='black', label=r'''∫f(x)dx''')
plt.plot(h_, trapezoid_sum, color='violet', label=r'''Формула трапеции''')
plt.xlabel('$h$', fontsize=10)
plt.ylabel(r'''$∫f(x)dx$''', fontsize=10)
plt.legend(loc='lower left')
plt.grid(True)


# случай, когда используется формула Симпсона
sp3 = plt.subplot(133)
plt.plot(h_, I_, color='black', label=r'''∫f(x)dx''')
plt.plot(h_, Simpson_s_sum, color='pink', label=r'''Формула Симпсона''')
plt.xlabel('$h$', fontsize=10)
plt.ylabel(r'''$∫f(x)dx$''', fontsize=10)
plt.legend(loc='lower left')
plt.grid(True)

plt.show()