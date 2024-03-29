#%%
# 
import pypsa
import numpy as np
import pandas as pd 
import os
#%%
try :
    network = pypsa.Network('data/networks/elec_s_37_ec_lcopt_Co2L.nc')
except :
    os.chdir('..')
    network = pypsa.Network('data/networks/elec_s_37_ec_lcopt_Co2L.nc')


#%%

n_days = 40

days = pd.Series(np.arange(24/3,8760/3+24/3,24/3))

days = days.sample(n_days,random_state=42)
days = days.sort_index()
#days = days.reset_index(drop=True)

new_snapshots = []
for idx in days:
    new_snapshots += list(network.snapshots[idx-24:idx])

network.snapshots = pd.DatetimeIndex(new_snapshots)
network.snapshot_weightings[:] = n_days/8760





#%%

n_new = 8759

r = np.random.rand(len(network.snapshots))

r_sorted = r.copy()
r_sorted.sort()

filt = r<r_sorted[n_new]
# %%

network.snapshots = network.snapshots[filt]
#network.snapshot_weightings[filt] = 8760/n_new

#%%

for g in network.generators.query('carrier == "OCGT"').index:
    network.generators.loc[g,'p_nom_extendable'] = False

#%%

ocgt_marginal = 58.3846
ocgt_capital = 47234
ocgt_efficiency = 0.39

for bus in network.buses.index:
    network.add('Generator',f'{bus} OCGT_e',
                bus=bus,
                carrier='OCGT',
                p_nom_extendable=True,
                marginal_cost = ocgt_marginal,
                capital_cost = ocgt_capital)


# %%

network.lopf(solver_name='gurobi',pyomo=False,formulation='kirchhoff',solver_options=dict(method=2,crossover=0))
# %%

network.export_to_netcdf('data/networks/elec_s_37_ec_lcopt_Co2L_h{}.nc'.format(n_new))
# %%



