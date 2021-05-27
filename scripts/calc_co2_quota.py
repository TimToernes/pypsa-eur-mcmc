#%%
import pypsa
from _helpers import configure_logging
import numpy as np
from pypsa.linopt import get_var, define_constraints, linexpr, get_dual
import os 
import matplotlib.pyplot as plt


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



# %%

networks_e= []

path_list = os.listdir('../inter_results/sweep_e_2030/')
path_list.sort()

for p in path_list:
    n = pypsa.Network('../inter_results/sweep_e_2030/'+ p)
    networks_e.append(n)

networks_h = []

path_list = os.listdir('../inter_results/sweep_H_2030/')
path_list.sort()

for p in path_list:
    n = pypsa.Network('../inter_results/sweep_H_2030/'+ p)
    networks_h.append(n)
# %%


base_emis_e = 1481895952
base_emis_h = 2206933437.8055553

co2_red_e = 1-np.array([n.global_constraints.constant['CO2Limit'] for n in networks_e])/base_emis_e
co2_emis_e = np.array([n.global_constraints.constant['CO2Limit'] for n in networks_e])

co2_price_e = np.array([n.global_constraints.mu['CO2Limit'] for n in networks_e])


co2_red_h = 1-np.array([n.global_constraints.constant['CO2Limit'] for n in networks_h])/base_emis_h
co2_emis_h = np.array([n.global_constraints.constant['CO2Limit'] for n in networks_h])
co2_price_h = np.array([n.global_constraints.mu['CO2Limit'] for n in networks_h])


# %%

plt.plot(co2_red_e*100,co2_price_e,label='Elec CO2 price')
plt.plot(co2_red_h*100,co2_price_h,label='Heat+Elec CO2 price')

plt.hlines([50],xmin=min(co2_red_e)*100,xmax=100,colors='green',label='Est. CO2 price 2030')
plt.vlines(91.5,0,50,linestyles='dashed')
plt.vlines(66.5,0,50,linestyles='dashed',colors='orange')
plt.legend()
plt.ylim(0,200)
plt.xlabel('CO2 reduction [% since 1990 in the modelled sectors]')
plt.ylabel('CO2 price [€/T]')
plt.text(30,60,'Estimated CO2 price in 2030')
plt.savefig('../graphics/co2_price_heat.jpeg')

# %%

base_emis_h*(1-0.665)

# %%
