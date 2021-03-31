import geographiclib
import numpy as np
import nvector as nv
from nvector import rad, deg

# lat_EA, lon_EA, z_EA = rad(1), rad(2), 3
# lat_EB, lon_EB, z_EB = rad(1), rad(2), 11

# n_EA_E = nv.lat_lon2n_E(lat_EA, lon_EA)
# n_EB_E = nv.lat_lon2n_E(lat_EB, lon_EB)

# p_AB_E = nv.n_EA_E_and_n_EB_E2p_AB_E(n_EA_E, n_EB_E, z_EA, z_EB)
# R_EN = nv.n_E2R_EN(n_EA_E)
# p_AB_N = np.dot(R_EN.T, p_AB_E).ravel()


def lla2enu(init_lla,point_lla):
    n_EA_E = nv.lat_lon2n_E(init_lla[0], init_lla[1])
    n_EB_E = nv.lat_lon2n_E(point_lla[0], point_lla[1])
    p_AB_E = nv.n_EA_E_and_n_EB_E2p_AB_E(n_EA_E, n_EB_E, init_lla[2], point_lla[2])
    R_EN = nv.n_E2R_EN(n_EA_E)
    p_AB_N = np.dot(R_EN.T, p_AB_E).ravel()
    p_AB_N[0],p_AB_N[1] = p_AB_N[1],p_AB_N[0]
    return p_AB_N

init_lla = np.array([rad(47.5094),rad(6.79395),367.163])
point_lla = np.array([rad(47.5115),rad(6.79323),414.594])
print(lla2enu(init_lla,point_lla))
