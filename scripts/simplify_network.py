#%%
# 
import pypsa
import numpy as np
import os
#%%
try :
    network = pypsa.Network('data/networks/elec_s_37_ec_lcopt_Co2L.nc')
except :
    os.chdir('..')
    network = pypsa.Network('data/networks/elec_s_37_ec_lcopt_Co2L.nc')

#%%

n_new = 100

r = np.random.rand(len(network.snapshots))

r_sorted = r.copy()
r_sorted.sort()

filt = r<r_sorted[n_new]
# %%

network.snapshots = network.snapshots[filt]
network.snapshot_weightings[filt] = 8760/n_new
# %%

network.lopf(solver_name='gurobi',pyomo=False,formulation='kirchhoff',solver_options=dict(method=2,crossover=0))
# %%

network.export_to_netcdf('data/networks/elec_s_37_ec_lcopt_Co2L_h{}.nc'.format(n_new))
# %%
