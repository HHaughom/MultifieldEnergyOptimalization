# Number of variables and inputs
Nx = 1


# All value here in kW
P_GTG_MAX = 4500
P_MAX = 10000
P_DEL_MAX = 15000


# Objective costs parameters
q_s = 10
q_SOC = .01
q_CO2 = 1


# Physical parameters
battery_eta = 0.8
kappa = 2
Eb_max = 50000
eta_b = 0.80

alpha = 50               # Finn k denna egentlig e. Blir M_CO2/M_CH4/LHV_CH4 elns.
incline = 1.06393049    # Linear fit approx med y = A*x er dette A. Bruk i GTG optimizer tingen


H = 2000
rho_liq = 1
g = 9.81
eta_p = 1
Pp_max = 15000000
Q_max = Pp_max*eta_p / (H*rho_liq*g)