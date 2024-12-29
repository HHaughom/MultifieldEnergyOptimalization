# Number of state and inputs ------------------------
Nx_p = 8
Nu_p = 3

# Physical parameters ------------------------------
tau_V = 0.5
tau_Pl = 0.5

Pl_max = 4500

# GTG Efficiency curve fittig parameters
a1 = -0.5714           
a2 = 1.286             
a3 = 0.2757            


# Tuning parameters -------------------------------
r_dP = .1
r_dgamma = 1
q_Q = 50
q_P = 50
q_s = 100
#q_P = 1

# Constraint parameters ---------------------------------
Q_min = 0
#Q_max = 10000
Pl_min = 0
Pl_max = 4500
gamma_min = 0
gamma_max = 1
dQ_max = 800
dgamma_max = 0.2

# Initial values
Pp0 = 0
Q0 = 0
V0 = 0
Pl0 = 0
gamma0 = 0

# Scaling parameter for delivered power
Pdel_max = 15000


# Diverse nye for ny pumpe
H = 2000    # Head
eta_p = 1   # Efficiency of pump
rho_liq = 1 # Density of water
g = 9.81    # Gravity const. 
tau_Q = 2   # Time constant for pump

Pp_max = 15000      # 20 000 kW or 20MW
#dPp_max = 0.4       # Percent change from timestep to timestep! Burde nok vaere mindre men start der
dPp_max = 1