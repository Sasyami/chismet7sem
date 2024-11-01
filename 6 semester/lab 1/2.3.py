import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

h = 0.01
c = 0.5
a = 0.5

Ꚍ = c * h / a

x0 = 0.5
w = 10


def g(t):
    return np.arccos(np.cos(w * t)) ** 4 * 0.01


def 𐓑(x, x0, t):
    if x == x0: return g(t) / h
    else: return 0


x = np.arange(0.0, 5.0, h)

t = np.arange(0.0, 5.0, Ꚍ)


def difference_scheme(a, h, x, 𐓑, idx_t):
    u_current = np.zeros(len(x))
    u_tmp = np.zeros(len(x))

    for j in range(len(x)):
        u_current[j] = 0

    for n in range(0, int(idx_t)):
        for j in range(1, len(x)):
            u_tmp[j] = u_current[j] + Ꚍ * (𐓑(x[j], x0, t[n]) - a * (u_current[j] - u_current[j - 1]) / h)
        u_tmp[0] = 0

        for j in range(len(x)):
            u_current[j] = u_tmp[j]

    return u_current


def exact(a, x, idx_t):
    u = np.zeros(len(x))

    for j in range(len(x)):
        if 0 < x[j] - x0 < a * t[int(idx_t)]:
            u[j] = (1 / a) * g(t[int(idx_t)] + (x0 - x[j]) / a)
        else:
            u[j] = 0

    return u


def animation_u(idx_t):
    return difference_scheme(a, h, x, 𐓑, idx_t)


def animation_u_exact(idx_t):
    return exact(a, x, idx_t)


idx_t = 0

fig, ax = plt.subplots()
line1, = ax.plot(x, animation_u(idx_t), color='pink', lw=2, label=r'$h = 10^{-2}$')
line2, = ax.plot(x, animation_u_exact(idx_t), "--", color='black', lw=2, label='Exact')
plt.legend()
plt.ylim(-0.1, 3)
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
