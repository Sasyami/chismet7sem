import numpy as np
import matplotlib.pyplot as plt


# интерполируемая функция
def F(x):
    return np.sin(np.e ** (x / 2) / 35)


# задание интервала
x_beginning = 0
x_ending = 10

# a.)
# задание числа узлов
n_a = 30

# узлы интерполяции
x_a = np.linspace(x_beginning, x_ending, n_a)
# x*
_x_a = []
for k in range(1, n_a):
    _x_a.append((x_a[k - 1] + x_a[k]) / 2)
# шаг
h = (x_ending - x_beginning) / (n_a - 1)

# значения функции в узлах интерполяции
f_a = [F(i) for i in x_a]
# значения функции в x*
F_a = [F(i) for i in _x_a]

# б.)
# задание числа узлов Чебышева
n_b = n_a

# узлы интерполяции
x_b = []
for k in range(1, n_b + 1):
    x_b.append((1 / 2) * (x_beginning + x_ending) + (1 / 2) * (x_ending - x_beginning) * np.cos(
        (2 * k - 1) * np.pi / (2 * n_b)))
# x*
_x_b = []
for k in range(1, n_b):
    _x_b.append((x_b[k - 1] + x_b[k]) / 2)

# значения функции в узлах интерполяции
f_b = [F(i) for i in x_b]
# значения функции в x*
F_b = [F(i) for i in _x_b]


def Lagrange(x, f, x_):
    sum_ = 0
    for i in range(len(x)):
        product_ = 1
        for j in range(len(x)):
            if i != j:
                product_ = product_ * (x_ - x[j]) / (x[i] - x[j])
        sum_ = sum_ + f[i] * product_
    return sum_


figtext_args = (0.5, 0, "Узлами интерполяции являются: a.) равноотстоящие узлы, б.) узлы Чебышева")
figtext_kwargs = dict(horizontalalignment="center", fontsize=9, wrap=True)
plt.figtext(*figtext_args, **figtext_kwargs)
plt.suptitle("Численная интерполяция функции f(x)", fontsize=11)

# случай, когда узлами интерполяции являются равноотстоящие узлы
sp1 = plt.subplot(121)
plt.plot(_x_a, F_a, color='lightskyblue')
plt.plot(_x_a, [Lagrange(x_a, f_a, i) for i in _x_a], linestyle='--', marker='o', color='black',
         markerfacecolor='orchid')
plt.xlabel('$x$', fontsize=10)
plt.legend(('$f(x)$', '$L{}_n(x), n=30$'))
plt.grid(True)

# случай, когда узлами интерполяции являются узлы Чебышева
sp2 = plt.subplot(122)
plt.plot(_x_b, F_b, color='lightskyblue')
plt.plot(_x_b, [Lagrange(x_b, f_b, i) for i in _x_b], linestyle='--', marker='o', color='black',
         markerfacecolor='orchid')
plt.xlabel('$x$', fontsize=10)
plt.legend(('$f(x)$', '$L{}_n(x), n=30$'))
plt.grid(True)

plt.show()