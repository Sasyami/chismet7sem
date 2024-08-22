import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

h = 0.001
σ = 0.5
k = 1
Ꚍ = 0.01

α_L = 1
β_L = 3


def μ_L(t):
    return np.sqrt(2 + t)


α_R = 1
β_R = 3


def μ_R(t):
    return (t - 2) / np.sqrt(1 + t)


def Ⴔ(x):
    return np.sqrt(2 - x ** 2)


def f(x, t):
    return (6 + 3 * t - x ** 2) / (2 * (2 + t - x ** 2) ** (3 / 2))


x = np.arange(0.0, 1.0, h)
t = np.arange(0.0, 1.0, Ꚍ)

C0 = 3 * β_L / (2 * h * α_L - β_L)

B = np.zeros(len(x))
B[0] = -4 * β_L / (2 * h * α_L - β_L)
B[1] = Ꚍ * σ * (C0 + 1) / (Ꚍ * σ * (2 - B[0]) + h ** 2)
for j in range(2, len(x) - 1):
    B[j] = Ꚍ * σ / (Ꚍ * σ * (2 - B[j - 1]) + h ** 2)


def A_(idx, u):
    A = np.zeros(len(x))
    A[0] = 2 * h * μ_L(t[int(idx)]) / (2 * h * α_L - β_L)
    for j in range(1, len(x) - 1):
        A[j] = (Ꚍ * h ** 2 / (Ꚍ * σ * (2 - B[j - 1]) + h ** 2)) * (f(x[j], (t[int(idx) - 1] + t[int(idx)]) / 2) + (1 - σ) * (u[j + 1] - 2 * u[j] + u[j - 1]) / h ** 2 + u[j] / Ꚍ + A[j - 1] * σ / h ** 2)
    A[len(x) - 1] = (h * μ_R(t[int(idx)]) + 2 * β_R * A[len(x) - 2] - 0.5 * β_R * (A[len(x) - 3] + B[len(x) - 3] * A[len(x) - 2])) / (h * α_R + 1.5 * β_R - 2 * β_R * B[len(x) - 2] + 0.5 * β_R * B[len(x) - 2] * B[len(x) - 3])
    return A


def Krank_Nicholson_scheme(idx_t):
    u_current = np.zeros(len(x))
    u_tmp = np.zeros(len(x))

    for j in range(len(x)):
        u_current[j] = Ⴔ(x[j])

    for n in range(0, int(idx_t)):
        A = A_(n + 1, u_current)

        u_tmp[len(x) - 1] = A[len(x) - 1]
        for j in range(len(x) - 2, 0, -1):
            u_tmp[j] = A[j] + B[j] * u_tmp[j + 1]
        u_tmp[0] = A[0] + B[0] * u_tmp[1] + C0 * u_tmp[2]

        for j in range(len(x)):
            u_current[j] = u_tmp[j]

    return u_current


def exact(idx_t):
    u = np.zeros(len(x))

    for j in range(len(x)):
        u[j] = np.sqrt(2 + t[int(idx_t)] - x[j] ** 2)

    return u


def animation_u(idx_t):
    return Krank_Nicholson_scheme(idx_t)


def animation_u_exact(idx_t):
    return exact(idx_t)


idx_t = 0

fig, ax = plt.subplots()
line1, = ax.plot(x, animation_u(idx_t), color='pink', lw=2, label=r'$h = 10^{-2}$')
line2, = ax.plot(x, animation_u_exact(idx_t), "--", color='black', lw=2, label='Exact')
plt.legend()
ax.set_xlabel('x', rotation=0)
ax.set_ylabel('u', rotation=0, labelpad=10)
plt.ylim(1, 2)
ax.set_title('Схема Кранка—Николсон. Время t = 0.0.')

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
    line1.set_ydata(animation_u(t_slider.val))
    line2.set_ydata(animation_u_exact(t_slider.val))
    ax.set_title('Схема Кранка—Николсон. Время t = {}.'.format(t[int(t_slider.val)]))
    fig.canvas.draw_idle()


t_slider.on_changed(update)

plt.show()