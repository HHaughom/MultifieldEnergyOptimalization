from casadi import *
from platform_parameters import *
from icecream import ic
from typing import Dict


def gtg_eta(Pl_scaled):
    return a1*(Pl_scaled)**2 + a2*(Pl_scaled) + a3


def gtg_P(Pl):
    """
    Returns output power from GTG unscaled with input scaled
    """
    return gtg_eta(Pl) * Pl * Pl_max



def pump_Q(P):
    """
    Returns flowrate of pump given power input. Power unscaled
    """
    return P*eta_p / (H*rho_liq*g)


def pump_P(Q):
    return H*rho_liq*g*Q / eta_p


def gtg_dynamics() -> Dict:
    V = MX.sym('V')
    Pl = MX.sym('Pl')
    gamma = MX.sym('gamma')
    dgamma = MX.sym('dgamma')

    V_dot = (gamma - V) / tau_V
    Pl_dot = (V - Pl) / tau_Pl
    gamma_dot = dgamma

    return {
        'x': vertcat(V, Pl, gamma),
        'x_dot': vertcat(V_dot, Pl_dot, gamma_dot),
        'u': dgamma
    }


def pump_dynamics() -> Dict:
    Pp = MX.sym('Pp')
    Q = MX.sym('Q')
    dPp = MX.sym('dPp')

    Pp_dot = dPp
    Q_dot = (Pp - Q) / tau_Q

    return {
        'x': vertcat(Pp, Q),
        'x_dot': vertcat(Pp_dot, Q_dot),
        'u': dPp
    }


def platform_dynamics() -> Dict:
    gtgs = [gtg_dynamics(), gtg_dynamics()]
    pump = pump_dynamics()

    x = vertcat(pump['x'], gtgs[0]['x'], gtgs[1]['x'])
    x_dot = vertcat(pump['x_dot'], gtgs[0]['x_dot'], gtgs[1]['x_dot'])
    u = vertcat(pump['u'], gtgs[0]['u'], gtgs[1]['u'])

    return {
        'x': x,
        'x_dot': x_dot,
        'u': u,
        'f': Function('f', [x, u], [x_dot], ['x', 'u'], ['x_dot'])
    }


def platform_integrator(T :int, N :int, finite_el :int) -> Dict:
    dt = T/N
    intg_options = {
        "simplify": True,
        "number_of_finite_elements": finite_el
    }

    dynamics = platform_dynamics()
    x = dynamics['x']
    u = dynamics['u']
    f = dynamics['f']

    # DAE instantiation
    dae = {
        'x': x,
        'p': u,     # Assume control input is constant during integration
        'ode': f(x,u)
    }

    # Integrator that integrates the system dynamics one step (dt timestep forward) using runge-kutta
    intg = integrator('F', 'rk', dae, 0, dt, intg_options)

    # Accessing the x-values after integration
    res = intg(x0=x, p=u)
    x_next = res['xf']

    # Simplifying API to go from (x,u) -> x_next
    F = Function('F', [x, u], [x_next], ['x', 'u'], ['x_next'])

    # Accumulating F to go from (x,u) -> x_next to (x,u) -> (x_trajectory, u_inputs)
    sim = F.mapaccum(N)


    return {
        'F': F,
        'sim': sim
    }


if __name__ == '__main__':
    ic(platform_dynamics())
    ic(platform_integrator(10, 20, 10))