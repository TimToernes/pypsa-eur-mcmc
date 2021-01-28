#%%
import pypsa
import logging
from _helpers import configure_logging
import numpy as np
from pypsa.linopt import get_var, define_constraints, linexpr
import re 
import os
import shutil
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from _mcmc_helpers import get_theta,calc_co2_emis_pr_node,read_csv
#%%


def draw_theta(theta,sigma,upper_bound=1,lower_bound=-1,max_iter=10000):
    count = 0
    f_inv = lambda x : np.tan(x)
    f = lambda x: np.tanh(x)    
    while count<max_iter:

        tan_theta = f_inv(theta)
        tan_theta_proposed=np.random.multivariate_normal(tan_theta,sigma)
        theta_proposed = f(tan_theta_proposed)
        if not (any(theta_proposed>1) or any(theta_proposed<-1)):
            break
        count += 1
    else :
        print('max iter reached')
        print(count)
    return theta_proposed



def increment_sample(path):
    folder,file = os.path.split(path)
    sample_str = re.search('s([0-9]?[0-9]?[0-9]?[0-9])',file).group()
    current_sample = int(sample_str[1:])
    file_new = file.replace(sample_str,'s{}'.format(current_sample+1))
    path_out = os.path.join(folder, file_new )
    return path_out


def extra_functionality(network, snapshots,variables,local_emis ):
    # Local emisons constraints 
    for i, bus in enumerate(variables):
        vars = []
        constants = []
        for t in network.snapshots: 
            for gen in network.generators.query('bus == {}'.format(str(bus))).index:
                vars.append(get_var(network,'Generator','p').loc[t,gen])
                const = 1/network.generators.efficiency.loc[gen] 
                const *= network.snapshot_weightings.loc[t]
                const *= network.carriers.co2_emissions.loc[network.generators.carrier.loc[gen]]

                constants.append(const)

        expr = linexpr((constants,vars)).sum()
        define_constraints(network,expr,'<=',local_emis[i],'local_co2','bus {}'.format(i))


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


def calc_pr(network,cost):
    # Calculate probability estimator, based on solution cost
    cost_0 = network.objective_optimum 
    mga_constraint_fullfilment = (cost-cost_0)/cost_0/network.mga_slack
    Pr = lambda c :(-2/(1+np.exp(-10*c+10))+2)
    pr_i = Pr(mga_constraint_fullfilment)
    return pr_i

def calc_capital_cost(network,mcmc_variables,theta_proposed):
    # Calculate the minimum capital cost of producing the proposed solution
    cost = 0
    for i in range(len(theta_proposed)):
        try :
            cost += min(network.generators.capital_cost.loc[mcmc_variables[i]] * theta_proposed[i])
        except : 
            cost += min(network.storage_units.capital_cost.loc[mcmc_variables[i]] * theta_proposed[i])
    # add line cost
    cost += sum(network.links.p_nom_min * network.links.capital_cost)
    return cost

def sample(network,alpha):

    mcmc_variables = read_csv(network.mcmc_variables)
    theta = get_theta(network,mcmc_variables)
    #sigma = np.array(read_csv(snakemake.input[1])).astype(float)
    sigma = np.genfromtxt(snakemake.input[1])


    #Take step
    theta_proposed = draw_theta(theta,sigma)
    #

    # Evaluate capital costs of the proposed solution (theta_porposed) and use this as early rejection criteria
#    cost_early = calc_capital_cost(network,mcmc_variables,theta_proposed)
#    pr_early = calc_pr(network,cost_early)

#    if pr_early>alpha: # Sample not rejected based on early evaluation
        # Evaluate network at new point 
    theta_base = np.genfromtxt(network.theta_base)
    co2_alocations = theta_base - theta_proposed*theta_base

    network = solve_network(network,mcmc_variables,co2_alocations)
    cost_i = network.objective 
    pr_i = calc_pr(network,cost_i)
#   else : # Sample rejected based on early evaluation 
 #       logging.info('Early rejection')
 #       pr_i = pr_early

    return network, pr_i


#%%
if __name__=='__main__':
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        try:
            snakemake = mock_snakemake('run_single_chain',chain=0,sample=6)
            os.chdir('..')
        except :
            
            os.chdir('..')
            snakemake = mock_snakemake('run_single_chain',chain=0,sample=6)

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
        alpha = np.random.rand()
        # Get the likelihood estimate pr_i from the network 
        network, pr_i = sample(network,alpha)
        # Accept or reject the sample 
        if alpha<pr_i: # Sample accepted, save solved network
            logging.info('sample accepted')
            network.sample = n_sample+1
            network.accepted = 1
            network.export_to_netcdf(out)
        else : # Sample rejected, copy previous network to next
            logging.info('sample rejected')
            network = pypsa.Network(inp)
            network.sample = n_sample+1
            network.accepted = 0 
            network.export_to_netcdf(out)
            #shutil.copyfile(inp,out)

        # Increment file names and sample number
        n_sample = n_sample +1
        inp = increment_sample(inp)
        out = increment_sample(out)


# %%

#theta = np.array([1,-0.4])

#sigma = np.array([[0.5,0],[0,0.5]])

# %%
