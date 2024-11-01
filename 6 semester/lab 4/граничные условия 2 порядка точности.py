import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

h = 0.1
c = 0.7
ρ = 1
m = ρ * h
T = 5

a = np.sqrt(T / ρ)
Ꚍ = c * h / a

q = 10
E0 = 2

l = 20
x0 = l / 2
ω = np.pi * a / l

t0 = 1
A = 2


def f(t):
    return q * E0 * np.sin(ω * t) / T


x = np.arange(0.0, l, h)
x1 = x[x <= x0]
x2 = x[x >= x0]

t = np.arange(0.0, 20.0, Ꚍ)


def cross_scheme(idx_t):
    u1_prev = np.zeros(len(x1))
    u1_current = np.zeros(len(x1))
    u1_tmp = np.zeros(len(x1))
    u1_add = np.zeros(len(x1))

    u2_prev = np.zeros(len(x2))
    u2_current = np.zeros(len(x2))
    u2_tmp = np.zeros(len(x2))
    u2_add = np.zeros(len(x2))

    for j in range(len(x1)):
        u1_prev[j] = 0
    for j in range(len(x2)):
        u2_prev[j] = 0

    for n in range(0, int(idx_t)):

        if n == 0:
            for j in range(len(x1)):
                u1_current[j] = 0
            for j in range(len(x2)):
                u2_current[j] = 0
        if n == 1:
            for j in range(1, len(x1) - 1):
                u1_tmp[j] = 2 * u1_current[j] - u1_prev[j] + (Ꚍ ** 2 * a ** 2 / (h ** 2)) * (u1_current[j + 1] - 2 * u1_current[j] + u1_current[j - 1])
            u1_tmp[0] = 0

            for j in range(1, len(x2) - 1):
                u2_tmp[j] = 2 * u2_current[j] - u2_prev[j] + (Ꚍ ** 2 * a ** 2 / (h ** 2)) * (u2_current[j + 1] - 2 * u2_current[j] + u2_current[j - 1])
            u2_tmp[len(x2) - 1] = 0

            u1_tmp[len(x1) - 1] = (Ꚍ ** 2 * (-u2_tmp[2] + 4 * u2_tmp[1] + 4 * u1_tmp[len(x1) - 2] - u1_tmp[len(x1) - 3]) + 2 * h * (m / T) * (2 * u1_current[len(x1) - 1] - u1_prev[len(x1) - 1]) + 2 * h * Ꚍ ** 2 * f(t[n + 1])) / (6 * Ꚍ ** 2 + 2 * h * (m / T))
            u2_tmp[0] = u1_tmp[len(x1) - 1]
        else:
            for j in range(1, len(x1) - 1):
                u1_add[j] = 2 * u1_tmp[j] - u1_current[j] + (Ꚍ ** 2 * a ** 2 / (h ** 2)) * (u1_tmp[j + 1] - 2 * u1_tmp[j] + u1_tmp[j - 1])
            u1_add[0] = 0

            for j in range(1, len(x2) - 1):
                u2_add[j] = 2 * u2_tmp[j] - u2_current[j] + (Ꚍ ** 2 * a ** 2 / (h ** 2)) * (u2_tmp[j + 1] - 2 * u2_tmp[j] + u2_tmp[j - 1])
            u2_add[len(x2) - 1] = 0

            u1_add[len(x1) - 1] = (Ꚍ ** 2 * (-u2_add[2] + 4 * u2_add[1] + 4 * u1_add[len(x1) - 2] - u1_add[len(x1) - 3]) + 2 * h * (m / T) * (5 * u1_tmp[len(x1) - 1] - 4 * u1_current[len(x1) - 1] + u1_prev[len(x1) - 1]) + 2 * h * Ꚍ ** 2 * f(t[n + 1])) / (6 * Ꚍ ** 2 + 4 * h * (m / T))
            u2_add[0] = u1_add[len(x1) - 1]

            for j in range(len(x1)):
                u1_prev[j] = u1_current[j]
                u1_current[j] = u1_tmp[j]
                u1_tmp[j] = u1_add[j]
            for j in range(len(x2)):
                u2_prev[j] = u2_current[j]
                u2_current[j] = u2_tmp[j]
                u2_tmp[j] = u2_add[j]

    if idx_t == 0:
        return np.concatenate((u1_prev, u2_prev[1:]))
    if idx_t == 1:
        return np.concatenate((u1_current, u2_current[1:]))
    else: return np.concatenate((u1_tmp, u2_tmp[1:]))


def animation(idx_t):
    return cross_scheme(idx_t)


idx_t = 0

fig, ax = plt.subplots()
plt.plot(x, np.zeros_like(x), color='lightgrey', lw=1)
line1, = ax.plot(x, animation(idx_t), color='black', lw=2, label=r'$h = 10^{-2}$')
plt.legend()
ax.set_xlabel('x', rotation=0)
ax.set_ylabel('u', rotation=0, labelpad=10)
plt.ylim(-70, 50)
ax.set_title('Решение задачи 2.28. Время t = 0.0.')

fig.subplots_adjust(left=0.1, bottom=0.25)

t_ax = fig.add_axes([0.1, 0.1, 0.8, 0.03])
t_slider = Slider(
    ax=t_ax,
    label='idx_t',
    valmin=0.0,
    valmax=len(t) - 1,
    valinit=idx_t,
    color='black',
    initcolor='black'
)


def update(val):
    line1.set_ydata(animation(t_slider.val))
    ax.set_title('Решение задачи 2.28. Время t = {}.'.format(t[int(t_slider.val)]))
    fig.canvas.draw_idle()


t_slider.on_changed(update)

plt.show()