import torch
import torch.linalg
import numpy as np

def rk4(x, h, f):

    k1 = f(x)
    k2 = f(x + h * 0.5 * k1)
    k3 = f(x + h * 0.5 * k2)
    k4 = f(x + h * 1.0 * k3)

    return x + h * (k1/6 + k2/3 + k3/3 + k4/6)


def explicit_euler(x, h, f):

    return x + h * f(x)


def explicit_midpoint(x, h, f):

    k1 = f(x)
    k2 = f(x + 0.5*h*k1)

    return x + h * k2


def implicit_euler(x, h, f, eps=1e-4, max_it=40):

    y = x + h * f(x)
    z = x + h * f(y)

    i = 0
    while torch.linalg.norm(y - z) > eps:
        if i > max_it:
            raise RuntimeError("Implicit Euler step not converging.")

        i += 1
        y = z
        z = x + h * f(y)

    return y


def implicit_midpoint(x, h, f, eps=1e-4, max_it=40):

    fx = f(x)
    y = x + h * fx
    z = x + h * (0.5*f(y) + 0.5*fx)

    i = 0
    while torch.linalg.norm(y - z) > eps:
        if i > max_it:
            print(z)
            raise RuntimeError("Implicit midpoint step not converging.")

        i += 1
        y = z
        z = x + h * (0.5*f(y) + 0.5*fx)

    return y



def integrate_model(model, times, x0, rk_method=rk4):

    if len(x0.shape) == 1: # If x0 is not a batch, expand the tensor, integrate, then collapse.
        sol = torch.zeros(size=(1, len(times), x0.shape[-1]))
        x = x0[None,...]

    else: # Assuming first axis is batch.
        sol = torch.zeros(size=(x0.shape[0], len(times), *x0.shape[1:]))
        x = x0
    
    rk_step = lambda x, h: rk_method(x, h, model)

    hs = times[1:] - times[:-1]

    sol[:,0,...] = x
    for i, h in enumerate(hs, start=1):
        x = rk_step(x, h)
        sol[:,i,...] = x
        
    if len(x0.shape) == 1: # If x0 is not a batch
        sol = sol[0,...]

    return sol


def integrate_model_training(model, x0, step_size, times, rk_method=rk4):
    """
        model:  a subclass of torch.nn.Module with implemented model.forward()-method.
        x0:     the initial condition of the ODE.
        step_size: the step size to be used.
        times:  the times the state is to be saved and output for. These are the pred_times
                in the trajectory dataset.
        rk_method: a function of signature (x, h, f, (eps, max_it)) returning a single step
                of a Runge-Kutta method starting in x, with step size h, and function the 
                callable f.
    """

    x = x0
    
    rk_step = lambda x, h: rk_method(x, h, model)

    sol = torch.zeros(size=(x0.shape[0], len(times), *x0.shape[1:]))
    """First axis is trajectory, second axis is time."""
    
    t = 0.0
    for i, t_next in enumerate(times):

        next_time = False
        while not next_time:
            if t_next - t < 1.1*step_size:
                h = t_next - t
                t = t_next
                x = rk_step(x, h)
                sol[:,i,...] = x
                next_time = True
            
            else:
                h = step_size
                t = t + step_size
                x = rk_step(x, h)

    return sol

