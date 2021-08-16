#%%
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
import pickle
import pandas as pd 
from solutions import solutions
import multiprocessing as mp


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

if __name__ == '__main__':
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        if not 'Snakefile' in  os.listdir():
            os.chdir('..')
        snakemake = mock_snakemake('co2_aloc_scenarios')

    builtins.snakemake = snakemake



    network = pypsa.Network(snakemake.input.network, 
                            override_component_attrs=override_component_attrs)
    network = set_link_locations(network)

    cts = network.buses.country.unique()
    cts = cts[:33]

    co2_totals = pd.read_csv('data/co2_totals.csv',index_col=0)

    co2_targets = pd.read_csv('data/co2_targets.csv',index_col=0)

    co2_base_emis = co2_totals.loc[cts, "electricity"].sum()

    co2_red = snakemake.config['co2_budget']/(co2_base_emis*1e6)

    national_co2 = co2_totals.loc[cts, "electricity"]

    local_1990 = national_co2*co2_red*1e6

    load = network.loads_t.p.groupby(network.buses.country,axis=1).sum().sum()

    local_load = load/sum(load)*co2_base_emis*co2_red*1e6


    # Emission share from elec sector assumed to be 35 %, based on 
    # calculation below 
    # co2_totals.loc['EU28']['electricity']/co2_totals.loc['EU28'].sum()

    # Total emission reductions relative to 1990 is only expected to be 
    # 30% according to the following calculation
    # (co2_targets[1990].sum()-co2_targets['2030 targets'].sum())/co2_targets[1990].sum()


    EU_ETS_country_share = co2_targets['2030 targets']/co2_targets['2030 targets'].sum()

    eu_ets_target = EU_ETS_country_share * co2_base_emis* (1-0.30) * 1e6 

    # For the countries not part of the EU ETS, set the allowable emissions 
    # to be a high number such that the constraint is not binding
    i_non_ets = set(cts).difference(set(EU_ETS_country_share.index))
    i_non_model = set(EU_ETS_country_share.index).difference(set(cts))

    for c in i_non_ets:
        eu_ets_target[c] = co2_base_emis*1e6 

    for c in i_non_model:
        eu_ets_target.pop(c)

    emis_alloc_schemes = {'local_load':local_load,'local_1990':local_1990,'optimum':None,'eu_ets_2018':eu_ets_target}


    for emis_alloc in emis_alloc_schemes:
        
        country_emis = emis_alloc_schemes[emis_alloc]

        network.global_constraints.constant['CO2Limit'] = snakemake.config['co2_budget']
        if emis_alloc == 'optimum':
            snakemake.config['use_local_co2_constraints'] = False
            
        else : 
            try :
                country_emis['EU']
            except :
                country_emis['EU'] = 0
            snakemake.config['use_local_co2_constraints'] = True
            snakemake.config['local_emission_constraints'] = country_emis
        
        network = set_link_locations(network)
        network = solve_network.solve_network(network)

        p = f'inter_results/{snakemake.config["run_name"]}/network_{co2_red*100:.0f}_{emis_alloc}.nc'

        if not os.path.exists(f'inter_results/{snakemake.config["run_name"]}/'):
            os.mkdir(f'inter_results/{snakemake.config["run_name"]}/')

        network.name = os.path.relpath(os.path.normpath(p))
        network.dual_path = network.name[:-2]+'p'
        duals = network.dualvalues
        pickle.dump((network.duals,network.dualvalues),open(network.dual_path, "wb" ))
        network.export_to_netcdf(p)

        try : 
            sol.put(network)
        except Exception:
            man = mp.Manager()
            sol = solutions(network, man)



    sol.merge()

    try :
        sol.save_csv(f'results/{snakemake.config["run_name"]}/result_')
    except Exception:
        os.mkdir(f'results/{snakemake.config["run_name"]}')
        sol.save_csv(f'results/{snakemake.config["run_name"]}/result_')



# %%
