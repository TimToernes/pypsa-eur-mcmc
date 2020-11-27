#%%
import pypsa
import logging
import pandas as pd
from _helpers import configure_logging
import sys
import time 
import os
import csv
import numpy as np 

#%%

def write_csv(path,item):
    # Write a list or numpy array (item) as csv file 
    # to the specified path
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(item)


def calc_variables(network):
    # Given a pypsa network the function will calculate the mcmc_variables 
    # Currently the variables are all extendable generators and extendable storage units 
    wind_filt = [carrier == 'offwind-ac' or carrier == 'offwind-dc' or carrier == 'onwind' for carrier in network.generators.carrier]

    network.generators.loc[wind_filt,'carrier'] = 'wind'


    variables = []
    for bus,_ in network.buses.iterrows():
        # Append generators
        q_str = 'bus == "{}" and p_nom_extendable == True'.format(bus)
        groups = network.generators.query(q_str).groupby(network.generators.carrier).groups
        variables.extend([list(value) for key,value in groups.items()])
        # Append storage units 
        q_str = 'bus == "{}" and p_nom_extendable == True'.format(bus)
        groups = network.storage_units.query(q_str).groupby(network.storage_units.carrier).groups
        variables.extend([list(value) for key,value in groups.items()])
    return variables

#%%
if __name__=='__main__':
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        try:
            snakemake = mock_snakemake('initialize_networks')
        except :
            import os
            os.chdir(os.getcwd()+'/scripts')
            snakemake = mock_snakemake('initialize_networks')
            os.chdir('..')
        
    configure_logging(snakemake)

    network = pypsa.Network(snakemake.input.network)

    # Store parameters in the network object 
    network.mga_slack = snakemake.config.get('mga_slack')
    
    mcmc_variables = calc_variables(network)
    network.mcmc_variables = "results/mcmc_variables.csv"
    write_csv(network.mcmc_variables,mcmc_variables)

    sigma = np.identity(len(mcmc_variables))*snakemake.config['sampler']['eps']
    network.sigma = "inter_results/sigma_s1.csv"
    np.savetxt(network.sigma,sigma)
    # write_csv(network.sigma,sigma)

    #network.generators['p_nom_extendable'] = True 
    
    # solve network to get optimum solution
    network.lopf(**snakemake.config.get('solver'))
    network.objective_optimum = network.objective

    # Save the starting point for each chain
    for i,p in enumerate(snakemake.output[:-1]):

        network.name = os.path.relpath(os.path.normpath(p))
        network.chain = i
        network.sample = 1
        network.export_to_netcdf(p)


# %%

# %%
