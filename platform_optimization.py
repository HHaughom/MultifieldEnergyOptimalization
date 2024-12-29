from platform_dynamics import platform_integrator, gtg_P, pump_P
from casadi import *
from platform_parameters import *
from icecream import ic


def create_platform_MPC(T :int, N :int, finite_el :int=1):
    F = platform_integrator(T, N, finite_el)['F']

    opti = Opti()
    x = opti.variable(Nx_p, N)
    u = opti.variable(Nu_p, N-1)
    s = opti.variable(N)  # Slack variable for energy balance

    Qd = opti.parameter()
    Pdel = opti.parameter()
    x0 = opti.parameter(Nx_p)
    
    set_objective(opti, x, u, s, Qd, N)
    set_constraints(opti, F, x, u, Pdel, s, x0, N)
    config_solver(opti)

    return {
        'opti': opti,
        'intg': F,
        'x': x,
        'u': u,
        'x0': x0,
        'Qd': Qd,
        'Pdel': Pdel,
        's': s
    }


def set_objective(opti, x, u, s, Qd, N):
    cost_x = 0
    cost_u = 0
    cost_s = 0
    R = diag(MX([r_dP, r_dgamma, r_dgamma]))

    for i in range(N):
        if i < N-1:
            cost_u += u[:,i].T @ R @ u[:,i]

        cost_Q = q_Q*(x[1,i] - Qd)**2

        P1 = gtg_P(x[3,i]) / Pl_max
        P2 = gtg_P(x[6,i]) / Pl_max
        cost_P  = q_P*(x[3,i]-P1)**2 + q_P*(x[6,i]-P2)**2

        cost_s += q_s*s[i]**2
        
        cost_x += cost_Q + cost_P + cost_s

    opti.minimize(cost_x + cost_u)
    #opti.minimize(cost_Q + cost_s)

def set_constraints(opti, F, x, u, Pdel, s, x0, N):
    """
    Setting constraints on the optimization problem for the MPC
    """
    # Initial condition
    opti.subject_to( x[:,0] == x0 )

    # Dynamics
    for i in range(N-1):
        opti.subject_to( x[:, i+1] == F(x[:,i], u[:,i]) )

    # State and input constraints
    for i in range(N):
        opti.subject_to( opti.bounded(0, x[0,i], 1) )
        opti.subject_to( opti.bounded(0, x[1,i], 1) )
        opti.subject_to( opti.bounded(0, x[2,i], 1) )
        opti.subject_to( opti.bounded(0, x[3,i], 1) )
        opti.subject_to( opti.bounded(0, x[4,i], 1) )
        opti.subject_to( opti.bounded(0, x[5,i], 1) )
        opti.subject_to( opti.bounded(0, x[6,i], 1) )
        opti.subject_to( opti.bounded(0, x[7,i], 1) )

        if i < N-1:
            opti.subject_to( opti.bounded(-dPp_max, u[0,i], dPp_max) )
            opti.subject_to( opti.bounded(-dgamma_max, u[1,i], dgamma_max) )
            opti.subject_to( opti.bounded(-dgamma_max, u[2,i], dgamma_max) )
            

    # Energy balance
    for i in range(N):
        P1      = gtg_P(x[3,i])
        P2      = gtg_P(x[6,i])
        Ppump   = x[0,i]*Pp_max
        opti.subject_to( P1 + P2 + Pdel*Pdel_max  == Ppump + s[i])

def config_solver(opti):
    p_opts = {'print_time': False, 'verbose': False, 'error_on_fail': True}
    #p_opts = {}
    s_opts = {'print_level': 0}
    opti.solver("ipopt", p_opts, s_opts)

    #opts = {'error_on_fail': True}
    #opti.solver('ipopt', opts)      




if __name__ == '__main__':
    ic(create_platform_MPC(500, 100, 10, 20, 10))