import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
from matplotlib.widgets import Slider, Button

N = 180
steps = 1000
dt = 0.5

q_init, f_init, nu_init = 0.002, 1.2, 0.55
Du_init, Dv_init = 0.18, 0.09
reseeding_interval, reseeding_zones = 55, 7
num_levels = 13

np.random.seed(42)

def add_perturbations(U, V, n=8):
    for _ in range(n):
        rad = np.random.randint(8, 16)
        cx, cy = np.random.randint(rad, N - rad, 2)
        y, x = np.ogrid[:N, :N]
        mask = (x - cx) ** 2 + (y - cy) ** 2 < rad ** 2
        U[mask] = 0.37 + 0.11 * np.random.rand(np.sum(mask))
        V[mask] = 0.28 + 0.10 * np.random.rand(np.sum(mask))

def get_new_state():
    U = np.ones((N, N)) + 0.01 * np.random.randn(N, N)
    V = np.zeros((N, N)) + 0.01 * np.random.randn(N, N)
    add_perturbations(U, V, n=8)
    return U, V

U, V = get_new_state()

def petri_mask(N):
    y, x = np.ogrid[:N, :N]
    return (x - N // 2) ** 2 + (y - N // 2) ** 2 <= (N // 2 - 6) ** 2

circle_mask = petri_mask(N)

def laplacian(Z):
    return (
        -4 * Z
        + np.roll(Z, +1, axis=0)
        + np.roll(Z, -1, axis=0)
        + np.roll(Z, +1, axis=1)
        + np.roll(Z, -1, axis=1)
    )

fig = plt.figure(figsize=(18, 10))
fig.patch.set_facecolor('white')
gs = fig.add_gridspec(8, 28)
ax_sim = fig.add_subplot(gs[:, :13])
ax_sim.set_facecolor('white')
ax_sim.axis('off')
ax_sim.set_title("Belousov-Zhabotinsky Reaction", fontsize=16, color='crimson')

ax_green = fig.add_subplot(gs[0:2, 15:27])
ax_yellow = fig.add_subplot(gs[2:4, 15:27])
ax_blue = fig.add_subplot(gs[4:6, 15:27])
ax_purple = fig.add_subplot(gs[6:8, 15:27])

plot_axes = [ax_green, ax_yellow, ax_blue, ax_purple]
plot_colors = ['forestgreen', 'gold', 'dodgerblue', 'purple']
param_names = ['<U>', 'std(U)', '<V>', 'max(U) - min(U)']

for ax, val, col in zip(plot_axes, param_names, plot_colors):
    ax.set_xlim(0, steps - 1)
    ax.set_ylabel(val, color=col, fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', labelcolor=col)
    ax.grid(True, linestyle=':', alpha=0.6)
ax_purple.set_xlabel("Frame", fontsize=12)
plt.subplots_adjust(left=0.04, right=0.98, top=0.96, bottom=0.21, wspace=0.41, hspace=0.7)

cmap = plt.colormaps['Reds'].resampled(num_levels)
levels = np.linspace(0, 1.1, num_levels + 1)
im = ax_sim.imshow(np.full_like(U, np.nan), cmap=cmap, vmin=0, vmax=levels[-2], interpolation="none")
contour_set = [None]
lines = []
for color, ax in zip(plot_colors, plot_axes):
    line, = ax.plot([], [], color=color)
    lines.append(line)
xdata = []
plot_data = [[] for _ in range(4)]
frame_counter = [0]

slider_color = 'whitesmoke'
slider_w = 0.06
slider_h = 0.03
slider_bottom = 0.06
sliders = []
slider_labels = ['q', 'f', 'nu', 'Du', 'Dv']
slider_args = [
    (0.0005, 0.008, q_init, 'tab:red'),
    (0.6, 2.0, f_init, 'tab:orange'),
    (0.4, 0.8, nu_init, 'tab:green'),
    (0.04, 0.35, Du_init, 'tab:cyan'),
    (0.01, 0.17, Dv_init, 'tab:blue')
]
param_vals = {
    'q': q_init, 'f': f_init, 'nu': nu_init, 'Du': Du_init, 'Dv': Dv_init
}

for i, (label, (vmin, vmax, vinit, color)) in enumerate(zip(slider_labels, slider_args)):
    ax_slider = fig.add_axes([0.08 + i * (slider_w + 0.08), slider_bottom, slider_w, slider_h], facecolor=slider_color)
    slider = Slider(ax_slider, label, vmin, vmax, valinit=vinit, color=color)
    param_vals[label] = vinit
    sliders.append(slider)

reset_ax = fig.add_axes([0.76, slider_bottom, 0.12, slider_h])
reset_button = Button(reset_ax, 'Reset', color='lightcoral', hovercolor='lightpink')

def sliders_changed(val):
    for label, slider in zip(slider_labels, sliders):
        param_vals[label] = slider.val

def reset_pressed(event):
    global U, V, xdata, plot_data, frame_counter
    U, V = get_new_state()
    xdata.clear()
    for pdata in plot_data:
        pdata.clear()
    frame_counter[0] = 0
    for line in lines:
        line.set_data([], [])
    frame = np.zeros_like(U)
    frame[circle_mask] = U[circle_mask]
    frame[~circle_mask] = np.nan
    im.set_array(frame)
    if contour_set[0] is not None:
        contour_set[0].remove()
        contour_set[0] = None
    plt.draw()

for slider in sliders:
    slider.on_changed(sliders_changed)
reset_button.on_clicked(reset_pressed)

def dynamic_ylim(data, margin=0.1):
    if len(data) == 0:
        return (0, 1)
    dmin = np.nanmin(data)
    dmax = np.nanmax(data)
    if dmin == dmax:
        return (dmin - 0.5, dmax + 0.5)
    rng = dmax - dmin
    return (dmin - margin * rng, dmax + margin * rng)

def dynamic_xlim(xdata, window):
    if len(xdata) < window:
        return (0, window - 1)
    else:
        return (xdata[-window], xdata[-1])

window = 250

def animate(i):
    global U, V, xdata, plot_data, frame_counter
    q, f, nu, Du, Dv = [param_vals[x] for x in slider_labels]
    if frame_counter[0] >= steps:
        return [im] + lines
    if frame_counter[0] == 0:
        xdata.clear()
        for pdata in plot_data:
            pdata.clear()
    if frame_counter[0] % reseeding_interval == 0 and frame_counter[0] > 0:
        add_perturbations(U, V, n=reseeding_zones)
    U_lap = laplacian(U)
    V_lap = laplacian(V)
    dU = (U - U**2 - f * V * (U - q) / (U + q)) + Du * U_lap
    dV = (U - V) + Dv * V_lap
    U += dU * dt
    V += dV * dt
    U[:] = np.clip(U, 0, 1.5)
    V[:] = np.clip(V, 0, 1.5)
    U_smooth = gaussian_filter(U, sigma=1.2)
    U_disc = np.digitize(U_smooth, levels)
    U_band = levels[U_disc - 1].reshape(U_smooth.shape)
    frame = np.zeros_like(U_band)
    frame[circle_mask] = U_band[circle_mask]
    frame[~circle_mask] = np.nan
    im.set_array(frame)
    if contour_set[0] is not None:
        contour_set[0].remove()
    cs = ax_sim.contour(
        frame, levels=np.linspace(0, levels[-2], 12),
        colors='white', linewidths=1.15, alpha=0.99
    )
    contour_set[0] = cs

    mean_U = np.nanmean(U[circle_mask])
    std_U = np.nanstd(U[circle_mask])
    mean_V = np.nanmean(V[circle_mask])
    range_U = np.nanmax(U[circle_mask]) - np.nanmin(U[circle_mask])

    xdata.append(frame_counter[0])
    new_plot_series = [mean_U, std_U, mean_V, range_U]
    for k, val in enumerate(new_plot_series):
        plot_data[k].append(val)
        lines[k].set_data(xdata, plot_data[k])
        ylims = dynamic_ylim(plot_data[k])
        plot_axes[k].set_ylim(ylims)
        xlims = dynamic_xlim(xdata, window)
        plot_axes[k].set_xlim(xlims)
    frame_counter[0] += 1
    return [im] + lines

ani = FuncAnimation(fig, animate, frames=steps, interval=44, blit=False, repeat=True)
plt.show()
