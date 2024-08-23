import numpy as np
import matplotlib.pyplot as plt


# дифференцируемая функция
def F(x):
    return np.e ** (-x) * np.sin(x)


# производная функции
def F_derivative(x):
    return np.e ** (-x) * (np.cos(x) - np.sin(x))


fault_array_forward = []


def ln_max_err_forward(x, h):
    # правая разность
    forward_dif = []

    for i in range(len(x)):
        if i == len(x) - 1:
            forward_dif.append((F(x[i]) - F(x[i - 1])) / h)  # для крайней правой точки считаем значение производной через левую разность, поскольку порядок точности одинаков
        else:
            forward_dif.append((F(x[i + 1]) - F(x[i])) / h)  # стандартная формула для правой разности

    err = [abs(F_derivative(x[j]) - forward_dif[j]) for j in range(len(x))]
    fault_array_forward.append(np.log(max(err)))


fault_array_central = []


def ln_max_err_central(x, h):
    # центральная разность
    central_dif = []

    for i in range(len(x)):
        if i == 0:
            central_dif.append((-3 * F(x[i]) + 4 * F(x[i + 1]) - F(x[i + 2])) / (2 * h))  # для крайней левой точки
        elif i == len(x) - 1:
            central_dif.append((3 * F(x[i]) - 4 * F(x[i - 1]) + F(x[i - 2])) / (2 * h))  # для крайней правой точки
        else:
            central_dif.append((F(x[i + 1]) - F(x[i - 1])) / (2 * h))  # стандартная формула для центральной разности

    err = [abs(F_derivative(x[j]) - central_dif[j]) for j in range(len(x))]
    fault_array_central.append(np.log(max(err)))


# задание интервала
x_beginning = -0.8
x_ending = 0.8

# задание числа узлов сетки
n = 10

ln_h = []

for i in range(10):
    # узлы
    x = np.linspace(x_beginning, x_ending, n)

    # шаг
    h = (x_ending - x_beginning) / (n - 1)

    ln_max_err_forward(x, h)
    ln_max_err_central(x, h)

    ln_h.append(np.log(h))

    n *= 2

tg = np.abs(fault_array_central[0]-fault_array_central[1])/np.abs(ln_h[0] - ln_h[1])
print(tg)

plt.suptitle("Оценка порядка точности метода конечных разностей", fontsize=11)

# случай, когда используется правая разность
sp1 = plt.subplot(121)
plt.plot(ln_h, fault_array_forward, color='black')
plt.xlabel('ln(h)', fontsize=10)
plt.ylabel('ln(max_err)', fontsize=10)
plt.grid(True)

# случай, когда используется центральная разность
sp2 = plt.subplot(122)
plt.plot(ln_h, fault_array_central, color='black')
plt.xlabel('ln(h)', fontsize=10)
plt.ylabel('ln(max_err)', fontsize=10)
plt.grid(True)

plt.show()

