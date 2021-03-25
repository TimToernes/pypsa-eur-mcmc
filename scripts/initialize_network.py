"""
This module will do the following:
1) Load the pypsa network
2) Create the file "results/mcmc_variables.csv" containing the names of the mcmc variables
3) Solve the network and store the initial objective value in the network.objective_optimum
4) Save a network file for each mcmc chain using the naming scheme inter_results/network_c#_s#.nc
Here c is the chain number and s the sample number. s will always be one as it is the first sample
that is saved
"""

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

def calc_theta_base(network,co2_intensity = 4):  
    theta_base = []
    load_p = dict(network.loads_t.p.sum())
    for var in mcmc_variables:
        if network.generators.index.isin(var).any():
            theta_base.append(network.generators.p_nom_opt.loc[network.generators.index.isin(var)].sum())
        elif network.storage_units.index.isin(var).any() :
            theta_base.append(network.storage_units.p_nom_opt.loc[network.storage_units.index.isin(var)].sum())
        else :
            theta_base.append(sum([load_p[v] for v in var])*co2_intensity)

    return theta_base



def calc_variables(network):
    """ Given a pypsa network the function will calculate the names of the mcmc_variables """

    variables = []
    country_codes = list(set(network.buses.country))

    for code in country_codes:
        variables.append(list(network.buses.query('country == "{}"'.format(code)).index))

    return variables


def calc_150p_coal_emis(network,emis_factor=1.5):
    # Calculate the alowable emissions, if countries are constrained to not emit more co2 than 
    # the emissions it would take to cover 150% of the country demand with coal power 

    # data source https://ourworldindata.org/grapher/carbon-dioxide-emissions-factor
    # 403.2 kg Co2 pr MWh
    co2_emis_pr_ton = 0.45 # ton emission of co2 pr MWh el produced by coal
    country_loads = network.loads_t.p.groupby(network.buses.country,axis=1).sum()
    country_alowable_emis = country_loads.mul(network.snapshot_weightings,axis=0).sum()*co2_emis_pr_ton*emis_factor

    return country_alowable_emis



def set_link_locations(network):
    network.links['location'] = ""

    query_string = lambda x : f'bus0 == "{x}" | bus1 == "{x}" | bus2 == "{x}" | bus3 == "{x}" | bus4 == "{x}"'
    id_co2_links = network.links.query(query_string('co2 atmosphere')).index

    country_codes = network.buses.country.unique()
    country_codes = country_codes[:-1]

    # Find all busses assosiated with the model countries 
    country_buses = {code : [] for code in country_codes}
    for country in country_codes:
        country_nodes = list(network.buses.query('country == "{}"'.format(country)).index)
        for bus in country_nodes:
            country_buses[country].extend(list(network.buses.query('location == "{}"'.format(bus)).index))

    # Set the location of all links connection to co2 atmosphere 
    for country in country_buses:
        for bus in country_buses[country]:
            idx = network.links.loc[id_co2_links].query(query_string(bus))['location'].index
            #idx = network.links.query(query_string(bus))['location'].index
            network.links.loc[idx,'location'] = country

    # Links connecting to co2 atmosphere without known location are set to belong to EU
    idx_homeless = network.links.query(query_string('co2 atmosphere')).query('location == ""').index
    network.links.loc[idx_homeless,'location'] = 'EU'
    return network


#%%
if __name__ == '__main__':

    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        try:
            snakemake = mock_snakemake('initialize_networks')
            #os.chdir('..')
        except :

            os.chdir('..')
            snakemake = mock_snakemake('initialize_networks')
            #os.chdir('..')

    #configure_logging(snakemake)
    builtins.snakemake = snakemake

    # Copy config file to results folder 
    copyfile(snakemake.config['configfile'],snakemake.output[-1])

    network = pypsa.Network(snakemake.input.network, 
                            override_component_attrs=override_component_attrs)
    network = set_link_locations(network)
    network.global_constraints.constant=snakemake.config['co2_budget']
    # Store parameters in the network object
    network.mga_slack = snakemake.config.get('mga_slack')

    #mcmc_variables = calc_variables(network)
    mcmc_variables = network.buses.country.unique()
    mcmc_variables[np.where(mcmc_variables == '')] = 'EU'
    network.mcmc_variables = f"results/{snakemake.config['run_name']}/mcmc_variables.csv"
    write_csv(network.mcmc_variables,mcmc_variables)

    #sigma = np.identity(len(mcmc_variables))*float(snakemake.config['sampler']['eps'])
    sigma = np.ones(len(mcmc_variables))*float(snakemake.config['sampler']['eps'])
    network.sigma = f"inter_results/{snakemake.config['run_name']}/sigma_s1.csv"
    np.savetxt(network.sigma,sigma)
    
    #theta_base = calc_theta_base(network,co2_intensity=3)
    #network.theta_base = "inter_results/theta_base.csv"
    #np.savetxt(network.theta_base,theta_base)

    #network = solve_network.prepare_network(network)

    allowable_emis = calc_150p_coal_emis(network,)
    allowable_emis['EU'] = np.inf

    snakemake.config['use_local_co2_constraints'] = True
    snakemake.config['local_emission_constraints'] = allowable_emis


    # solve network to get optimum solution
    network = solve_network.solve_network(network)

    duals = network.dualvalues
    #network.lopf(**snakemake.config.get('solver'))
    network.objective_optimum = network.objective
    network.accepted = 1 

    co2_budget = snakemake.config['co2_budget']
    country_emis = get_country_emis(network)
    try :
        country_emis['EU']
    except :
        country_emis['EU'] = 0

    theta = np.array([country_emis[v] for v in mcmc_variables])/co2_budget
    network.theta = theta_to_str(theta)
    network.export_to_netcdf(f"results/{snakemake.config['run_name']}/network_c0_s1.nc")

    # Save the starting point for each chain
    for i,p in enumerate(snakemake.output[:-2]):

        network.name = os.path.relpath(os.path.normpath(p))
        network.chain = i
        network.sample = 1
        network.export_to_netcdf(p)

# %%

