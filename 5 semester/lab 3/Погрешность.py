import numpy as np
import matplotlib.pyplot as plt


# дифференцируемая функция
def F(x):
    return 1 / (2 * x ** 2 + 1)


# задание интервала
x_beginning = -1
x_ending = 1

# вычисленное значение интеграла
I = np.sqrt(2) * np.arctan(np.sqrt(2))


# формула средних прямоугольников

def ln_err_rectangle_rule(x, h):
    rectangle_sum = 0
    for i in range(len(x) - 1):
        rectangle_sum += F((x[i] + x[i+1]) / 2) * h
    return np.log(abs(rectangle_sum - I))


# формула трапеции

def ln_err_trapezoid_formula(x, h):
    trapezoid_sum = 0
    for i in range(len(x) - 1):
        trapezoid_sum += ((F(x[i]) + F(x[i+1])) / 2) * h
    return np.log(abs(trapezoid_sum - I))


# формула Симпсона

def ln_err_Simpson_s_formula(x, h):
    Simpson_s_sum = 0
    for i in range(len(x) - 1):
        Simpson_s_sum += (h / 6) * (F(x[i]) + 4 * F((x[i] + x[i+1]) / 2) + F(x[i+1]))
    return np.log(abs(Simpson_s_sum - I))


# задание числа узлов сетки
n = 10

ln_h = []
ln_err_rectangle = []
ln_err_trapezoid = []
ln_err_Simpson = []

rectangle_sum = []
trapezoid_sum = []
Simpson_s_sum = []

for i in range(3):
    # узлы
    x = np.linspace(x_beginning, x_ending, n)

    # шаг
    h = (x_ending - x_beginning) / (n - 1)

    ln_err_rectangle.append(ln_err_rectangle_rule(x, h))
    ln_err_trapezoid.append(ln_err_trapezoid_formula(x, h))
    ln_err_Simpson.append(ln_err_Simpson_s_formula(x, h))

    ln_h.append(np.log(h))

    n *= 10

tg1 = np.abs(ln_err_rectangle[0]-ln_err_rectangle[1])/np.abs(ln_h[0] - ln_h[1])
tg2 = np.abs(ln_err_trapezoid[0]-ln_err_trapezoid[1])/np.abs(ln_h[0] - ln_h[1])
tg3 = np.abs(ln_err_Simpson[0]-ln_err_Simpson[1])/np.abs(ln_h[0] - ln_h[1])
print(tg1, tg2, tg3)


plt.suptitle("Исследование зависимости ошибки вычислений от шага сетки", fontsize=11)

# случай, когда используется формула средних прямоугольников
sp1 = plt.subplot(131)
plt.plot(ln_h, ln_err_rectangle, color='black')
plt.xlabel('ln(h)', fontsize=10)
plt.ylabel('ln(err)', fontsize=10)
plt.grid(True)

# случай, когда используется формула трапеции
sp2 = plt.subplot(132)
plt.plot(ln_h, ln_err_trapezoid, color='black')
plt.xlabel('ln(h)', fontsize=10)
plt.ylabel('ln(err)', fontsize=10)
plt.grid(True)

# случай, когда используется формула Симпсона
sp3 = plt.subplot(133)
plt.plot(ln_h, ln_err_Simpson, color='black')
plt.xlabel('ln(h)', fontsize=10)
plt.ylabel('ln(err)', fontsize=10)
plt.grid(True)

plt.show()