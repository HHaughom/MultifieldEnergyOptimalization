from wf_dynamics import wf_integrator
from wf_parameters import *
from icecream import ic
from casadi import *



def create_wf_MPC(T :int, N :int, finite_el :int=1):
    F = wf_integrator(T, N)['F']

    opti = Opti()
    x = opti.variable(Nx_w, N)
    u = opti.variable(Nu_w, N-1)
    x0 = opti.parameter(Nx_w)
    s = opti.variable(N)
    
    v = opti.parameter()               # Wind conditions
    setpoints = opti.parameter(5)       # [Pdist1d, Pdist2d, Pb1d, Pb2d, Pcharge_d] coming from scheduler

    set_objective(opti, x, u, N, setpoints, s)
    set_constraints(opti, F, x, u, x0, v, s, N)
    config_solver(opti)

    return {
            'opti': opti,
            'intg': F,
            'x': x,
            'u': u,
            'x0': x0,
            'setpoints': setpoints,
            'v': v
    }


def set_constraints(opti, F, x, u, x0, v, s, N):

    opti.subject_to( x[:,0] == x0 )

    for i in range(N-1):
        opti.subject_to( x[:,i+1] == F(x[:,i], u[:,i], v) )

    for i in range(N):
        opti.subject_to( opti.bounded(0, x[0,i], 1) )
        opti.subject_to( opti.bounded(0, x[1,i], 1) )
        opti.subject_to( opti.bounded(0, x[2,i], 1) )
        opti.subject_to( opti.bounded(0, x[3,i], 1) )
        opti.subject_to( opti.bounded(0, x[4,i], 1) )
        opti.subject_to( opti.bounded(0, x[5,i], 1) )
        opti.subject_to( opti.bounded(0, x[6,i], 1) )

    # Distribution of energy balance
    for i in range(N):
        opti.subject_to( x[0,i] == x[1,i] + x[2,i] + x[6,i] + s[i] )

    # Delivered below max
    for i in range(N):
        opti.subject_to( x[1,i] + x[4,i] <= Pdel_max/P_MAX )
        opti.subject_to( x[2,i] + x[5,i] <= Pdel_max/P_MAX )
        #pass

    for i in range(N-1):
        opti.subject_to( opti.bounded(-dP_max, u[0,i], dP_max) )
        opti.subject_to( opti.bounded(-dP_max, u[1,i], dP_max) )
        opti.subject_to( opti.bounded(-dP_max, u[2,i], dP_max) )
        opti.subject_to( opti.bounded(-dP_max, u[3,i], dP_max) )
        opti.subject_to( opti.bounded(-1, u[4,i], 1) )
        #pass


def set_objective(opti, x, u, N, setpoints, s):
    cost_x = 0
    cost_u = 0
    cost_s = 0
    R = diag(MX([r_dP, r_dP, r_dP, r_dP, r_dP]))

    for i in range(N-1):
        cost_u += u[:,i].T @ R @ u[:,i]

    for i in range(N):
        cost_del1 = q_del*((x[1,i] + x[4,i]) - (setpoints[0] + setpoints[2]))**2
        cost_del2 = q_del*((x[2,i] + x[5,i]) - (setpoints[1] + setpoints[3]))**2

        cost_wtg = q_wtg*(x[1,i] - setpoints[0])**2 + q_wtg*(x[2,i] - setpoints[1])**2
        cost_b = q_b*(x[4,i] - setpoints[2])**2 + q_b*(x[5,i] - setpoints[3])**2
        cost_charge = q_charge*(x[6,i] - setpoints[4])**2
        
        cost_x += cost_del1 + cost_del2 + cost_wtg + cost_b + cost_charge

        cost_s += q_s*s[i]**2

    opti.minimize(cost_x + cost_u + cost_s)



def config_solver(opti):
    opts = {
        #'ipopt.tol': 1e-5,
        #'ipopt.constr_viol_tol': 1e-4,
        #'ipopt.dual_inf_tol': 1e-2,
        #'ipopt.compl_inf_tol': 1e-3,
        'ipopt.acceptable_tol': 1e-3,
        'ipopt.acceptable_constr_viol_tol': 0.1,
        'ipopt.acceptable_compl_inf_tol': 0.1,
        'ipopt.acceptable_dual_inf_tol': 1e10,
        'ipopt.acceptable_iter': 5,
        'error_on_fail': True,
        #'ipopt.mu_strategy': 'monotone'
        'ipopt.print_level': 0,
        'print_time': False,
        'verbose': False

    }
    opti.solver('ipopt', opts)



if __name__ == "__main__":
    ic(create_wf_MPC(60, 20))

    
