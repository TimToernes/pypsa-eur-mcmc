#%%
import pypsa
import pandas as pd 
import numpy as np
import time
import multiprocessing as mp
import os 
from solutions import solutions

#%%
override_component_attrs = pypsa.descriptors.Dict({k : v.copy() for k,v in pypsa.components.component_attrs.items()})
override_component_attrs["Link"].loc["bus2"] = ["string",np.nan,np.nan,"2nd bus","Input (optional)"]
override_component_attrs["Link"].loc["bus3"] = ["string",np.nan,np.nan,"3rd bus","Input (optional)"]
override_component_attrs["Link"].loc["bus4"] = ["string",np.nan,np.nan,"4th bus","Input (optional)"]
override_component_attrs["Link"].loc["efficiency2"] = ["static or series","per unit",1.,"2nd bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["efficiency3"] = ["static or series","per unit",1.,"3rd bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["efficiency4"] = ["static or series","per unit",1.,"4th bus efficiency","Input (optional)"]
override_component_attrs["Link"].loc["p2"] = ["series","MW",0.,"2nd bus output","Output"]
override_component_attrs["Link"].loc["p3"] = ["series","MW",0.,"3rd bus output","Output"]
override_component_attrs["Link"].loc["p4"] = ["series","MW",0.,"4th bus output","Output"]


def set_n_snapshots(network,n):
    network.snapshots = network.snapshots[:n]
    network.snapshot_weightings[:] = 8760/n
    return network


def speed_tjek(network,nhours):

    network = set_n_snapshots(network,nhours)

    timer = time.time()

    network.lopf(**solver)

    time_spend = (time.time()-timer)/60

    print(f"{time_spend:.1f}min spend, on {nhours} snapshots")

    return network, time_spend



#%%
try : 
    network = pypsa.Network('data/networks/elec_s_37_lv1.5__Co2L0p55-3H-H-solar+p3-dist1_2030.nc',
                            override_component_attrs=override_component_attrs)  
except : 
    os.chdir('..')
    network = pypsa.Network('data/networks/elec_s_37_lv1.5__Co2L0p55-3H-H-solar+p3-dist1_2030.nc',
                            override_component_attrs=override_component_attrs)  
man = mp.Manager()
sol = solutions(network, man)


solver = {
  "solver_name": 'gurobi',
  "formulation": 'kirchhoff',
  "pyomo": False,
  "keep_shadowprices": True,
  "solver_options": {
    "threads": 4,
    "method": 2, # barrier
    "crossover": 0,
    "BarConvTol": 1.e-9,
    "FeasibilityTol": 1.e-6,
    "AggFill": 0,
    "PreDual": 0,
    }}


tjek_hours = [50,100,200,400,800,1600]#,2400,2920]

for nhours in tjek_hours:

    network = pypsa.Network('data/networks/elec_s_37_lv1.5__Co2L0p55-3H-H-solar+p3-dist1_2030.nc',
                        override_component_attrs=override_component_attrs)
    
    network, time_spend = speed_tjek(network,nhours)
    
    network.c = time_spend
    network.s = nhours

    sol.put(network)

sol.merge()

try :
    os.mkdir('results/speed_test')
except : 
    pass
sol.save_csv(f'results/speed_test/result_')
# %%

#import matplotlib.pyplot as plt

# 50 snapshots = 32s 
# 100 snapshots = 81,7s
# 200 snapshots = 174s 
# 400 snapshots = 15.8 min 
# 800 snapshots = 55.6min
# 1600 snapshots = 23 min 
# 2920 snapshots = 80 min

# 25 snapshots pr min

#x = [50,100,200,2920]
#y = [32,81,174,4800]

#plt.plot(x,y)
# %%
