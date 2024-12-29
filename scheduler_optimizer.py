from scheduler_dynamics import scheduler_integrator
from platform_dynamics import gtg_P, pump_P
from casadi import *
from scheduler_parameters import *
from icecream import ic


def create_scheduler_MPC(T :int, N :int, finite_el :int=1):

    F = scheduler_integrator(T, N)['F']

    opti = Opti()

    x = opti.variable(N)
    P_WTG = opti.variable(2, N-1)
    P_b = opti.variable(2, N-1)
    P_charge = opti.variable(N-1)
    P_l = opti.variable(4, N-1)
    Q = opti.variable(2, N-1)
    s = opti.variable(2)

    x0 = opti.parameter()
    P_expected = opti.parameter(N-1)
    V_target = opti.parameter(2)
    SOC_d = opti.parameter()

    set_objective(opti, x, P_l, SOC_d, s, N)
    set_constraints(opti, x, x0, F, Q, P_WTG, P_b, P_charge, s, P_expected, P_l, V_target, N)
    config_solver(opti)

    return {
        'opti': opti,
        'intg': F,
        'x': x,
        's': s,
        'x0': x0,
        'P_WTG': P_WTG,
        'P_charge': P_charge,
        'P_b': P_b,
        'P_expected': P_expected,
        'V_target': V_target,
        'SOC_d': SOC_d,
        'P_l': P_l,
        'Q': Q
    }



def set_objective(opti, x, P_l, SOC_d, s, N):
    cost_SOC = 0
    cost_CO2 = 0

    for i in range(N):
        cost_SOC += q_SOC*(x[i] - SOC_d)**2

    for i in range(N-1):
        cost_CO2 += q_CO2*(P_l[0,i]**2 + P_l[1,i]**2 + P_l[2,i]**2 + P_l[3,i]**2)

    cost_s = q_s*(s[0]**2 + s[1]**2)

    opti.minimize(cost_SOC + cost_CO2 + N**2*cost_s)


def set_constraints(opti, x, x0, F, Q, P_WTG, P_b, P_charge, s, P_expected, P_l, V_target, N):

    opti.subject_to( x[0] == x0 )

    # Dynamics constraints
    for i in range(N-1):
        opti.subject_to( x[i+1] == F(x[i], vertcat(P_charge[i], P_b[:,i])) )

    # Parameter constraints
    for i in range(N-1):
        opti.subject_to( P_expected[i] == P_WTG[0,i] + P_WTG[1,i] + P_charge[i] )

        # Bounds on all variables
        opti.subject_to( opti.bounded(0, P_WTG[0,i], 1) )
        opti.subject_to( opti.bounded(0, P_WTG[1,i], 1) )
        opti.subject_to( opti.bounded(0, P_b[0,i], 1) )
        opti.subject_to( opti.bounded(0, P_b[1,i], 1) )
        opti.subject_to( opti.bounded(0, Q[0,i], 1) )
        opti.subject_to( opti.bounded(0, Q[1,i], 1) )
        opti.subject_to( opti.bounded(0, P_charge[i], 1) )
        opti.subject_to( opti.bounded(0, P_l[0,i], 1) )
        opti.subject_to( opti.bounded(0, P_l[1,i], 1) )
        opti.subject_to( opti.bounded(0, P_l[2,i], 1) )
        opti.subject_to( opti.bounded(0, P_l[3,i], 1) )

        # Energy-balance on platforms
        opti.subject_to( pump_P(Q[0,i]*Q_max) / (P_MAX*1000) == ( gtg_P(P_l[0,i]) + gtg_P(P_l[1,i]) + (P_WTG[0,i] + P_b[0,i])*P_MAX ) / P_MAX )
        opti.subject_to( pump_P(Q[1,i]*Q_max) / (P_MAX*1000) == ( gtg_P(P_l[2,i]) + gtg_P(P_l[3,i]) + (P_WTG[1,i] + P_b[1,i])*P_MAX ) / P_MAX )

        # Max delivered to platforms
        opti.subject_to( (P_WTG[0,i] + P_b[0,i]) <= P_DEL_MAX / P_MAX )
        opti.subject_to( (P_WTG[1,i] + P_b[1,i]) <= P_DEL_MAX / P_MAX )

    for i in range(N):
        # Bounds on state of charge
        opti.subject_to( opti.bounded(0, x[i], 1) )

    # Pumping target constraints
    dt = 0.5        # Timestep in hours
    T = (N-1)*dt        # Prediction horizon
    V_max = T*Q_max*3600
    V1 = 0          # Volume pumped on platform1
    V2 = 0
    for i in range(N-1):
        V1 += Q[0,i]*Q_max * 1800
        V2 += Q[1,i]*Q_max * 1800   # Percentage of max * max pumpin in m3/s times how many seconds in a timestep

    V1 /= V_max
    V2 /= V_max    

    opti.subject_to( V_target[0] == V1 + s[0] )
    opti.subject_to( V_target[1] == V2 + s[1] )


def config_solver(opti):
    p_opts = {'print_time': False, 'verbose': False, 'error_on_fail': True}
    s_opts = {'print_level': 0}
    opti.solver("ipopt", p_opts, s_opts)

    #opts = {'error_on_fail': True}
    #opti.solver('ipopt', opts) 


if __name__ == '__main__':
    ic(create_scheduler_MPC(24, 48))