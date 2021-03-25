
#%% imports
from shutil import copyfile
import sys
import builtins 
import pypsa
import os
import csv
from _helpers import configure_logging
from _mcmc_helpers import *
import numpy as np
sys.path.append('./scripts/')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import solve_network
from solutions import solutions
import multiprocessing as mp
#import pandas as pd

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


#%%


def calc_150p_coal_emis(network,emis_factor=1.5):
    # Calculate the alowable emissions, if countries are constrained to not emit more co2 than 
    # the emissions it would take to cover 150% of the country demand with coal power 

    # data source https://ourworldindata.org/grapher/carbon-dioxide-emissions-factor
    # 403.2 kg Co2 pr MWh
    co2_emis_pr_ton = 0.45 # ton emission of co2 pr MWh el produced by coal
    country_loads = network.loads_t.p.groupby(network.buses.country,axis=1).sum()
    country_alowable_emis = country_loads.mul(network.snapshot_weightings,axis=0).sum()*co2_emis_pr_ton*emis_factor

    return country_alowable_emis


#%%
if __name__ == '__main__':

    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        try:
            snakemake = mock_snakemake('co2_sweep')
            #os.chdir('..')
        except :

            os.chdir('..')
            snakemake = mock_snakemake('co2_sweep')
            #os.chdir('..')

    #configure_logging(snakemake)
    builtins.snakemake = snakemake

    
    network = pypsa.Network(snakemake.input.network, 
                            override_component_attrs=override_component_attrs)

    
    allowable_emis = calc_150p_coal_emis(network,)
    allowable_emis['EU'] = np.inf

    snakemake.config['use_local_co2_constraints'] = True
    snakemake.config['local_emission_constraints'] = allowable_emis




    for co2_red in np.linspace(1,0,10):

        network.global_constraints.constant=snakemake.config['co2_budget']*co2_red
        # solve network to get optimum solution
        network = solve_network.solve_network(network)

        try : 
            sol.put(network)
        except Exception:
            man = mp.Manager()
            sol = solutions(network, man)

        p = f"inter_results/{snakemake.config['run_name']}/network_{co2_red:.2}.nc"

        network.export_to_netcdf(p)

    sol.save_csv(f'results/{snakemake.config["run_name"]}/result_')

        