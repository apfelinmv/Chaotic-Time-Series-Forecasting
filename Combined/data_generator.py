import numpy as np

def runge_kutta(time_steps, y0, system, params):
    ys = [y0]
    for t in range(len(time_steps) - 1):
        dt = time_steps[t + 1] - time_steps[t]
        k1 = system(time_steps[t],           y0,            params)
        k2 = system(time_steps[t] + dt/2,    y0 + dt/2*k1,  params)
        k3 = system(time_steps[t] + dt/2,    y0 + dt/2*k2,  params)
        k4 = system(time_steps[t] + dt,      y0 + dt*k3,    params)
        y0 = y0 + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        ys.append(y0)
    return np.array(ys)

def lorentz_ode(t, xyz, params):
    x, y, z = xyz
    σ, ρ, β = params['σ'], params['ρ'], params['β']
    dx = σ*(y - x)
    dy = x*(ρ - z) - y
    dz = x*y - β*z
    return np.array([dx, dy, dz])

def generate_lorenz_x():
    # задаём параметры
    time_steps = np.arange(0, 1500, 0.1)
    params     = {'σ':10., 'ρ':28., 'β':8/3}
    xyz0       = np.array([1.,1.,1.])
    sol        = runge_kutta(time_steps, xyz0, lorentz_ode, params)
    # берём хвост (после «прогрева») и нормируем только компоненту x
    x = sol[2000:,0]
    x = (x - x.min()) / (x.max() - x.min())
    return x
