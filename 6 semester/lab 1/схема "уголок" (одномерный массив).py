import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

h = 0.01
c = 0.5
a = 0.5

Ꚍ = c * h / a

x0 = 0.5
દ = 0.2


def ξ(x):
    return np.abs(x-x0) / દ


def Ⴔ1(x):
    return np.heaviside(1 - ξ(x), 1)


def Ⴔ2(x):
    return Ⴔ1(x) * (1 - ξ(x) ** 2)


def Ⴔ3(x):
    return Ⴔ1(x) * np.exp(- ξ(x) ** 2 / (np.abs(1 - ξ(x) ** 2)))


def Ⴔ4(x):
    return Ⴔ1(x) * (np.cos(np.pi * ξ(x) / 2) ** 3)


def 𐓑(x, t):
    return 0


x = np.arange(0.0, 5.0, h)

t = np.arange(0.0, 5.0, Ꚍ)


def difference_scheme(a, h, x, Ⴔ, 𐓑, μ, idx_t):
    u_current = np.zeros(len(x))
    u_tmp = np.zeros(len(x))

    for j in range(len(x)):
        u_current[j] = Ⴔ(x[j])

    for n in range(0, int(idx_t)):
        for j in range(1, len(x)):
            u_tmp[j] = u_current[j] + Ꚍ * (𐓑(x[j], t[n]) - a * (u_current[j] - u_current[j - 1]) / h)
        u_tmp[0] = μ(t[n + 1])

        for j in range(len(x)):
            u_current[j] = u_tmp[j]

    return u_current


def exact(a, x, Ⴔ, μ, idx_t):
    u = np.zeros(len(x))

    for j in range(len(x)):
        if x[j] >= a * t[int(idx_t)]:
            u[j] = Ⴔ(x[j] - a * t[int(idx_t)])
        else:
            u[j] = μ(t[int(idx_t)] - x[j] / a)

    return u


def animation_u(idx_t):
    return difference_scheme(a, h, x, Ⴔ1, 𐓑, Ⴔ2, idx_t)


def animation_u_exact(idx_t):
    return exact(a, x, Ⴔ1, Ⴔ2, idx_t)


idx_t = 0

fig, ax = plt.subplots()
line1, = ax.plot(x, animation_u(idx_t), color='pink', lw=2, label=r'$h = 10^{-2}$')
line2, = ax.plot(x, animation_u_exact(idx_t), "--", color='black', lw=2, label='Exact')
plt.legend()
ax.set_xlabel('x', rotation=0)
ax.set_ylabel('u', rotation=0, labelpad=10)
ax.set_title('Схема "уголок". Время t = 0.0.')

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
    ax.set_title('Схема "уголок". Время t = {}.'.format(t[int(t_slider.val)]))
    fig.canvas.draw_idle()


t_slider.on_changed(update)

plt.show()
