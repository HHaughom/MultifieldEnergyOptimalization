from casadi import *
from scheduler_parameters import *
from wf_dynamics import wtg_power
from icecream import ic 
from typing import Dict



def scheduler_dynamics() -> Dict:
    Pcharge = MX.sym('Pcharge')
    Pb1 = MX.sym('Pb1')
    Pb2 = MX.sym('Pb2')
    SOC = MX.sym('SOC')

    SOC_dot = (Pcharge*P_MAX * eta_b - (Pb1 + Pb2)*P_MAX) / (kappa * Eb_max)

    x = SOC
    x_dot = SOC_dot
    p = vertcat(Pcharge, Pb1, Pb2)

    return {
        'x': x,
        'x_dot': x_dot,
        'p': p,
        'f': Function('f', [x,p], [x_dot],  ['x','p'], ['x_dot'])
    }


def scheduler_integrator(T :int, N :int, finite_el :int=1) -> Dict:
    #dt = T/N
    dt = 0.5
    intg_options = {
        "simplify": True,
        "number_of_finite_elements": finite_el
    }

    dynamics = scheduler_dynamics()
    x = dynamics['x']
    x_dot = dynamics['x_dot']
    p = dynamics['p']
    f = dynamics['f']

     # DAE instantiation
    dae = {
        'x': x,
        'p': p,
        'ode': f(x,p)
    }

    # Integrator that integrates the system dynamics one step (dt timestep forward) using runge-kutta
    intg = integrator('F', 'rk', dae, 0, dt, intg_options)

    # Accessing the x-values after integration
    res = intg(x0=x, p=p)
    x_next = res['xf']

    # Simplifying API to go from (x,u) -> x_next
    F = Function('F', [x, p], [x_next], ['x', 'p'], ['x_next'])

    # Accumulating F to go from (x,u) -> x_next to (x,u) -> (x_trajectory, u_inputs)
    sim = F.mapaccum(N)

    return {
        'F': F,
        'sim': sim
    }




if __name__ == '__main__':
    ic(scheduler_integrator(24, 48))