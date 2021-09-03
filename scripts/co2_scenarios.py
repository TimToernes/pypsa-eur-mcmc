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
import time
from iso3166 import countries as iso_countries


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

def calc_150p_coal_emis(network,emis_factor=1.5):
    # Calculate the alowable emissions, if countries are constrained to not emit more co2 than 
    # the emissions it would take to cover 150% of the country demand with coal power 

    # data source https://ourworldindata.org/grapher/carbon-dioxide-emissions-factor
    # 403.2 kg Co2 pr MWh
    co2_emis_pr_ton = 0.45 # ton emission of co2 pr MWh el produced by coal
    country_loads = network.loads_t.p.groupby(network.buses.country,axis=1).sum()
    country_alowable_emis = country_loads.mul(network.snapshot_weightings,axis=0).sum()*co2_emis_pr_ton*emis_factor

    return country_alowable_emis

def create_country_pop_df(network):
    df_pop = pd.read_csv('data/API_SP.POP.TOTL_DS2_en_csv_v2_2106202.csv',
                        sep=',',
                        index_col=0,skiprows=3)
    df_gdp = pd.read_csv('data/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_2055594.csv',
                            sep=',',
                            index_col=0,skiprows=3)


    model_countries = network.buses.country.unique()[:33]
    alpha3 = [iso_countries.get(c).alpha3 for c in model_countries]
    df_pop_i = df_pop.set_index('Country Code')
    df_gdp_i = df_gdp.set_index('Country Code')

    model_countries_pop = pd.DataFrame(df_pop_i.loc[alpha3]['2019'])
    model_countries_gdp = pd.DataFrame(df_gdp_i.loc[alpha3]['2019'])
    model_countries_pop.index = model_countries
    model_countries_gdp.index = model_countries
    
    df_country_pop = pd.Series(model_countries_pop['2019'])
    df_country_gdp = pd.Series(model_countries_gdp['2019'])

    return df_country_pop, df_country_gdp

# %%

    # Emission share from elec sector assumed to be 35 %, based on 
    # calculation below 
    # co2_totals.loc['EU28']['electricity']/co2_totals.loc['EU28'].sum()

    # Total emission reductions relative to 1990 is only expected to be 
    # 30% according to the following calculation
    # (co2_targets[1990].sum()-co2_targets['2030 targets'].sum())/co2_targets[1990].sum()

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

    mcmc_variables = network.buses.country.unique()
    mcmc_variables[np.where(mcmc_variables == '')] = 'EU'

    co2_totals = pd.read_csv('data/co2_totals.csv',index_col=0)
    co2_targets = pd.read_csv('data/co2_targets.csv',index_col=0)
    co2_base_emis = co2_totals.loc[cts, "electricity"].sum()
    co2_budget = snakemake.config['co2_budget']
    co2_red = co2_budget/(co2_base_emis*1e6)

    df_country_pop, df_country_gdp = create_country_pop_df(network)

    national_co2 = co2_totals.loc[cts, "electricity"]

    emis_alloc_schemes = {}

    #### Local 1990 - Soverignity
    local_1990 = national_co2*co2_red*1e6
    emis_alloc_schemes['local_1990'] = local_1990


    #### Local load - 
    load = network.loads_t.p.groupby(network.buses.country,axis=1).sum().sum()
    local_load = load/sum(load)*co2_base_emis*co2_red*1e6
    local_load['EU'] = np.inf
    emis_alloc_schemes['local_load'] = local_load

    ### Optimal 
    allowable_emis = calc_150p_coal_emis(network,)
    allowable_emis['EU'] = np.inf
    emis_alloc_schemes['optimum'] = allowable_emis

    ### EU ETS
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
    eu_ets_target['EU'] = np.inf
    emis_alloc_schemes['eu_ets_2018'] = eu_ets_target

    ### Egalitarianism 
    emis_alloc_schemes['egalitarinism'] = df_country_pop/sum(df_country_pop) * co2_budget
    emis_alloc_schemes['egalitarinism']['EU'] = np.inf

    ### Ability to pay 
    # CO2 inversly proportional to gdp/pop
    rel_wealth = df_country_gdp/df_country_pop
    emis_alloc_schemes['ability_to_pay'] = (1/rel_wealth)/sum(1/rel_wealth) * co2_budget
    emis_alloc_schemes['ability_to_pay']['EU'] = np.inf

    ### Relative ability to pay - proportional to pop^2/gdp
    emis_alloc_schemes['rel_ability_to_pay'] = ((df_country_pop**2)/df_country_gdp)/sum(((df_country_pop**2)/df_country_gdp))*co2_budget
    emis_alloc_schemes['rel_ability_to_pay']['EU'] = np.inf

    ### Economic Activity 
    emis_alloc_schemes['economic_activity'] = df_country_gdp/sum(df_country_gdp)*co2_budget
    emis_alloc_schemes['economic_activity']['EU'] = np.inf

    ### Poluter pays 
    national_co2_no_ME = national_co2[national_co2.index != 'ME'] # ME is 0 in 1900 
    poluter_pays = (1/national_co2_no_ME)/sum(1/national_co2_no_ME)*co2_budget
    poluter_pays['ME'] = co2_budget



    
    #network.snapshots = network.snapshots[0:2]
    #network.snapshot_weightings = network.snapshot_weightings[0:2]

    for emis_alloc in emis_alloc_schemes:
        
        country_emis = emis_alloc_schemes[emis_alloc]

        
        if emis_alloc == 'optimum':
            snakemake.config['use_local_co2_constraints'] = True
            snakemake.config['local_emission_constraints'] = country_emis    
            network.global_constraints.constant['CO2Limit'] = snakemake.config['co2_budget']        
        else : 
            network.global_constraints.constant['CO2Limit'] = snakemake.config['co2_budget']*10
            
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

        if emis_alloc == 'optimum':
            country_emis = get_country_emis(network)
            theta = np.array([country_emis[v] for v in mcmc_variables[:-1]])/co2_budget

        else : 
            theta = np.array([country_emis[v] for v in mcmc_variables[:-1]])/co2_budget


        network.name = os.path.relpath(os.path.normpath(p))
        network.dual_path = network.name[:-2]+'p'
        duals = network.dualvalues
        pickle.dump((network.duals,network.dualvalues),open(network.dual_path, "wb" ))
        network.theta = theta_to_str(theta)
        network.export_to_netcdf(p)

        try : 
            sol.put(network)
        except Exception:
            man = mp.Manager()
            sol = solutions(network, man)

    time.sleep(10)
    sol.merge()

    try :
        sol.save_csv(f'results/{snakemake.config["run_name"]}/result_')
    except Exception:
        os.mkdir(f'results/{snakemake.config["run_name"]}')
        sol.save_csv(f'results/{snakemake.config["run_name"]}/result_')



# %%
