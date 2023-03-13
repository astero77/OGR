import torch
import numpy as np
from tqdm import tqdm
from torch.optim import Adam, SGD
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from Adam3 import AdamOGR

def rosenbrock(x):
    x_1 = torch.roll(x, -1, dims=0).float()[:-1 :]
    x = x.float()[:-1 :]

    return torch.sum(100 * (x_1 - x ** 2) ** 2 + (x - 1) ** 2, 0)

def run_optimization(xy_init, optimizer_class, n_iter, **optimizer_kwargs):
    xy_t = torch.tensor(xy_init, requires_grad=True)
    optimizer = optimizer_class([xy_t], **optimizer_kwargs)
    path = np.empty((n_iter + 1, xy_t.size(0)))
    path[0, :] = xy_init

    for i in tqdm(range(1, n_iter + 1)):
        optimizer.zero_grad()
        loss = rosenbrock(xy_t)
        #print(loss)
        # if loss < 1e-1:
        #     path[-1] = xy_t.detach().numpy()
        #     print (i)
        #     break
        loss.backward()
        torch.nn.utils.clip_grad_norm_(xy_t, 1.0)
        optimizer.step()

        path[i, :] = xy_t.detach().numpy()
    return path

if __name__ == "__main__":
    #print(rosenbrock(torch.tensor([1.0, 1.0])))
    xy_init = (0.9, -0.1, 0.9, 0.1, 0.04, 0.1, -0.5, -0.05, 0.03, 101.4, 0.2, -2.4, -0.1, 0.32, 0.1, 0.45, 0.01, -0.9, -12.9, 0, 0, 0.4)
    n_iter = 10000

    path_adam = run_optimization(xy_init, Adam, n_iter, betas=(0.9,0.999), lr=1e-1)
    #path_sgd = run_optimization(xy_init, SGD, n_iter, lr=1e-3)
    path_adamOGR = run_optimization(xy_init, AdamOGR, n_iter, lr=1, gamma=0.5)
    print(path_adam[-1], path_adamOGR[-1])
    

