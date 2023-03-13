import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Adam3 import AdamOGR

def rosenbrock(xy):
    x, y = xy

    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def run_optimization(xy_init, optimizer_class, n_iter, **optimizer_kwargs):

    xy_t = torch.tensor(xy_init, requires_grad=True)
    optimizer = optimizer_class([xy_t], **optimizer_kwargs)

    path = np.empty((n_iter + 1, 2))
    path[0, :] = xy_init

    for i in tqdm(range(1, n_iter + 1)):
        optimizer.zero_grad()
        loss = rosenbrock(xy_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(xy_t, 1.0)
        optimizer.step()

        path[i, :] = xy_t.detach().numpy()

    return path

def create_animation(paths,
                     colors,
                     names,
                     figsize=(12, 12),
                     x_lim=(-2, 2),
                     y_lim=(-1, 3),
                     n_seconds=5):

    if not (len(paths) == len(colors) == len(names)):
        raise ValueError

    path_length = max(len(path) for path in paths)

    n_points = 300
    x = np.linspace(*x_lim, n_points)
    y = np.linspace(*y_lim, n_points)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock([X, Y])

    minimum = (1.0, 1.0)

    fig, ax = plt.subplots(figsize=figsize)
    ax.contour(X, Y, Z, 90, cmap="jet")

    scatters = [ax.scatter(None,
                           None,
                           label=label,
                           c=c) for c, label in zip(colors, names)]

    ax.legend(prop={"size": 25})
    ax.plot(*minimum, "rD")

    def animate(i):
        for path, scatter in zip(paths, scatters):
            scatter.set_offsets(path[:i, :])

        ax.set_title(str(i))

    ms_per_frame = 1000 * n_seconds / path_length

    anim = FuncAnimation(fig, animate, frames=path_length, interval=ms_per_frame)

    return anim

if __name__ == "__main__":
    xy_init = (56.8, -23.2)
    n_iter = 2000

    path_adam = run_optimization(xy_init, Adam, n_iter, lr=1e-2)
    path_sgd = run_optimization(xy_init, SGD, n_iter, lr=1e-3)
    path_weird = run_optimization(xy_init, AdamOGR, n_iter, lr=1e-3)

    freq = 100

    paths = [path_adam[::freq], path_sgd[::freq], path_weird[::freq]]
    colors = ["green", "blue", "black"]
    names = ["Adam", "SGD", "Weird"]

    anim = create_animation(paths,
                            colors,
                            names,
                            figsize=(12, 7),
                            x_lim=(-60.01, 60.0),
                            y_lim=(-60.0, 60.0),
                            n_seconds=7)

    anim.save("result.gif")
    print(path_adam[-15:])
    print(path_weird[-15:])
    

