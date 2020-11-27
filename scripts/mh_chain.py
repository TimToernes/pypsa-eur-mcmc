#%%
import pypsa
import logging
from _helpers import configure_logging
import numpy as np
from pypsa.linopt import get_var, define_constraints, linexpr
import csv
import re 
import os
import shutil
#%%
def get_theta(netwrok,mcmc_variables):
    theta = []
    for var in mcmc_variables:
        if network.generators.index.isin(var).any():
            theta.append(network.generators.p_nom_opt.loc[network.generators.index.isin(var)].sum())
        else :
            theta.append(network.storage_units.p_nom_opt.loc[network.storage_units.index.isin(var)].sum())

    return theta


def read_csv(path):
    item = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            item.append(row)
    return item


def increment_sample(path):
    folder,file = os.path.split(path)
    sample_str = re.search('s([0-9]?[0-9]?[0-9]?[0-9])',file).group()
    current_sample = int(sample_str[1:])
    file_new = file.replace(sample_str,'s{}'.format(current_sample+1))
    path_out = os.path.join(folder, file_new )
    return path_out


def extra_functionality(network, snapshots, variables, theta):
    var_indexes_gen = get_var(network, 'Generator', 'p_nom').index
    var_indexes_stor = get_var(network, 'StorageUnit', 'p_nom').index
    for i in range(len(variables)):
        if any(var_indexes_gen.isin(variables[i])):
            var = get_var(network, 'Generator', 'p_nom').loc[var_indexes_gen.isin(variables[i])]
        elif any(var_indexes_stor.isin(variables[i])):
            var = get_var(network, 'StorageUnit', 'p_nom').loc[var_indexes_stor.isin(variables[i])]
        #print(var)
        expr = linexpr((1, var)).sum()
        #print(expr)
        #print(theta[i])
        define_constraints(network, expr, '==', theta[i], 'test_con', 'test{}'.format(i))

def solve_network(network,variables,theta):
    extra_func = lambda n, s: extra_functionality(n, 
                                                s, 
                                                variables, 
                                                theta)

    stat = network.lopf(**snakemake.config.get('solver'),
                        extra_functionality=extra_func)
    logging.info(stat)

    if str(stat[1])=='infeasible or unbounded':
        network.objective = np.inf
    return network

def sample(network):

    mcmc_variables = read_csv(network.mcmc_variables)
    theta = get_theta(network,mcmc_variables)
    #sigma = np.array(read_csv(snakemake.input[1])).astype(float)
    sigma = np.genfromtxt(snakemake.input[1])

    # Take step
    theta_proposed=np.random.multivariate_normal(theta,sigma)
    theta_proposed[theta_proposed<0]=0
    # Evaluate network at new point 
    network = solve_network(network,mcmc_variables,theta_proposed)
    
    # Evaluate quality of sample point 
    cost_i = network.objective
    cost_0 = network.objective_optimum 

    mga_constraint_fullfilment = (cost_i-cost_0)/cost_0/network.mga_slack

    Pr = lambda c :(-2/(1+np.exp(-10*c+10))+2)

    pr_i = Pr(mga_constraint_fullfilment)

    return network, pr_i

#%%
if __name__=='__main__':
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        try:
            snakemake = mock_snakemake('run_single_chain',chain=1,sample=201)
        except :
            
            os.chdir(os.getcwd()+'/scripts')
            snakemake = mock_snakemake('run_single_chain',chain=1,sample=201)
            os.chdir('..')
    configure_logging(snakemake,skip_handlers=True)

    # Get the sample number of the input network file
    pre,file = os.path.split(snakemake.input[0])
    sample_input = int(re.search('s([0-9]?[0-9]?[0-9])',file).group()[1:])

    # Sample itteratively until the desired sample number is reached
    n_sample = sample_input
    inp = snakemake.input[0]
    out = increment_sample(inp)
    while n_sample <= int(snakemake.wildcards.sample) :
    
        network = pypsa.Network(inp)
        # Get the likelihood estimate pr_i from the network 
        network, pr_i = sample(network)
        # Accept or reject the sample 
        if np.random.rand()<pr_i: # Sample accepted, save solved network
            logging.info('sample accepted')
            network.sample = n_sample
            network.export_to_netcdf(out)
        else : # Sample rejected, copy previous network to next
            logging.info('sample rejected')
            shutil.copyfile(inp,out)

        # Increment file names and sample number
        n_sample = n_sample +1
        inp = increment_sample(inp)
        out = increment_sample(out)


# %%




# %%
