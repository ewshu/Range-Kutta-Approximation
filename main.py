# Runge-Kutta (RK4) Numerical Integration for System of First-Order Differential Equations

import numpy as np
import matplotlib.pyplot as plt

def system_ode(_t, _y):
    """ Here is the system of first order differential equations.
    _t: discrete time step value
    _y: state vector [y1, y2] """
    return np.array([_y[1], -_t * _y[1] + (2 / _t) * _y[0]])

def rk4(func, tk, _yk, _dt=0.01, **addpar):
    """ Single-step fourth-order numerical integration (RK4) method in Python.
    func: system of first order ODEs
    tk: current time step
    _yk: current state vector [y1, y2, y3, ...]
    _dt: discrete time step size
    **addpar: additional parameters for ODE system
    returns: y evaluated at time k+1 """

    # Evaluating the derivative at several stages within time interval
    f1 = func(tk, _yk, **addpar)
    f2 = func(tk + _dt / 2, _yk + (f1 * (_dt / 2)), **addpar)
    f3 = func(tk + _dt / 2, _yk + (f2 * (_dt / 2)), **addpar)
    f4 = func(tk + _dt, _yk + (f3 * _dt), **addpar)

    # Returning an average of the derivative over tk, tk + dt
    return _yk + (_dt / 6) * (f1 + (2 * f2) + (2 * f3) + f4)

# Simulation harness

dt = 0.01
time = np.arange(1.0, 5.0 + dt, dt)

# Initial Conditions of a Second Order System [y1, y2] at t = 1
y0 = np.array([0, 1])

# Results of the Simulation
state_history = []

# Initialize yk
yk = y0

# Approximating y at t
for t in time:
    state_history.append(yk)
    yk = rk4(system_ode, t, yk, dt)

# Convering list to numpy array
state_history = np.array(state_history)

print(f'y evaluated at time t = {t} seconds: {yk[0]}')

# ==============================================================
# Plotting history on graph

fig, ax = plt.subplots()
ax.plot(time, state_history[:, 0])
ax.plot(time, state_history[:, 1])
ax.set(xlabel='t', ylabel='[Y]', title='Second Order System Propagation')
plt.legend(['y1', 'y2'])
ax.grid()
plt.show()
