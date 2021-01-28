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
import sys
import pypsa
import os
import csv
from _helpers import configure_logging
from _mcmc_helpers import write_csv
import numpy as np
sys.path.append('./scripts/')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
#import pandas as pd

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

    configure_logging(snakemake)

    network = pypsa.Network(snakemake.input.network)

    # Store parameters in the network object
    network.mga_slack = snakemake.config.get('mga_slack')

    mcmc_variables = calc_variables(network)
    network.mcmc_variables = "results/mcmc_variables.csv"
    write_csv(network.mcmc_variables,mcmc_variables)

    sigma = np.identity(len(mcmc_variables))*float(snakemake.config['sampler']['eps'])
    network.sigma = "inter_results/sigma_s1.csv"
    np.savetxt(network.sigma,sigma)
    
    theta_base = calc_theta_base(network,co2_intensity=3)
    network.theta_base = "inter_results/theta_base.csv"
    np.savetxt(network.theta_base,theta_base)
    

    # solve network to get optimum solution
    network.lopf(**snakemake.config.get('solver'))
    network.objective_optimum = network.objective
    network.accepted = 1 

    # Save the starting point for each chain
    for i,p in enumerate(snakemake.output[:-1]):

        network.name = os.path.relpath(os.path.normpath(p))
        network.chain = i
        network.sample = 1
        network.export_to_netcdf(p)

# %%


# %%
