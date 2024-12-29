from casadi import *
import numpy as np
from typing import Dict
from icecream import ic
from wf_parameters import *


def logistic(x, k, x0):
    return 1 / (1 + np.exp(-k * (x - x0)))

def wtg_power(v, v_min :float=5, v_max :float=26, k :float=5, k_max :float=10000, A :float=1, B :float=0.2) -> float:

    y1 = A * v**3
    y2 = k_max
    y3 = 0

    # Logistic blending
    smooth_y1 = y1 * (1 - logistic(v, 1, v_min))
    smooth_y2 = y2 * logistic(v, 1, v_min) * (1 - logistic(v, k, v_max))
    smooth_y3 = y3 * logistic(v, k, v_max)

    return (smooth_y1 + smooth_y2 + smooth_y3) / P_MAX


def wf_dynamics() -> Dict:
    Pcharge = MX.sym('Pcharge')
    Pb1 = MX.sym('Pb1')
    Pwtg1 = MX.sym('Pwtg1')
    Pb2 = MX.sym('Pb2')
    Pwtg2 = MX.sym('Pwtg2')
    SOC = MX.sym('SOC')
    Pv = MX.sym('Pv')
    v = MX.sym('v')

    dPwtg1 = MX.sym('dPwtg1')
    dPb1 = MX.sym('dPb1')
    dPwtg2 = MX.sym('dPwtg2')
    dPb2 = MX.sym('dPb2')
    dPcharge = MX.sym('dPcharge')

    Pv_dot = (wtg_power(v) - Pv) / tau_wtg
    SOC_dot = (Pcharge*eta_b - (Pb1 + Pb2))*P_MAX / (kappa*Eb_max)
    Pwtg1_dot = dPwtg1
    Pb1_dot = dPb1
    Pwtg2_dot = dPwtg2
    Pb2_dot = dPb2
    Pcharge_dot = dPcharge

    x = vertcat(Pv, Pwtg1, Pwtg2, SOC, Pb1, Pb2, Pcharge)
    x_dot = vertcat(Pv_dot, Pwtg1_dot, Pwtg2_dot, SOC_dot, Pb1_dot, Pb2_dot, Pcharge_dot)
    u = vertcat(dPwtg1, dPwtg2, dPb1, dPb2, dPcharge)
    p = v

    return {
        'x': x,
        'x_dot': x_dot,
        'u': u,
        'p': p,
        'f': Function('f', [x,u,p], [x_dot], ['x','u','p'], ['x_dot'])
    }


def wf_integrator(T :int, N :int, finite_el :int=1):
    dt = T/N
    
    intg_options = {
        "simplify": True,
        "number_of_finite_elements": finite_el
    }

    dynamics = wf_dynamics()
    x = dynamics['x']
    u = dynamics['u']
    f = dynamics['f']
    p = dynamics['p']

    # DAE instantiation
    dae = {
        'x': x,
        'u': u,   
        'p': p,
        'ode': f(x, u, p)
    }
    
    # Integrator that integrates the system dynamics one step (dt timestep forward) using runge-kutta
    intg = integrator('F', 'rk', dae, 0, dt, intg_options)

    # Accessing the x-values after integration
    res = intg(x0=x, u=u, p=p)
    x_next = res['xf']

    # Simplifying API to go from (x,u) -> x_next
    F = Function('F', [x, u, p], [x_next], ['x', 'u', 'p'], ['x_next'])

    # Accumulating F to go from (x,u) -> x_next to (x,u) -> (x_trajectory, u_inputs)
    sim = F.mapaccum(N)

    return {
        'F': F,
        'sim': sim
    }



if __name__ == '__main__':
    ic(wf_integrator(T=60, N=20))


