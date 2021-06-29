#%%
from numpy.core.numeric import ones
import pypsa
import logging
from _helpers import configure_logging
import numpy as np
from pypsa.linopt import get_var, define_constraints, linexpr
from pypsa.descriptors import free_output_series_dataframes
import re 
import os
import shutil
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from _mcmc_helpers import *
import solve_network
import pickle

#First tell PyPSA that links can have multiple outputs by
#overriding the component_attrs. This can be done for
#as many buses as you need with format busi for i = 2,3,4,5,....
#See https://pypsa.org/doc/components.html#link-with-multiple-outputs-or-inputs
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


def patch_pyomo_tmpdir(tmpdir):
    # PYOMO should write its lp files into tmp here
    import os
    if not os.path.isdir(tmpdir):
        os.mkdir(tmpdir)
    from pyutilib.services import TempfileManager
    TempfileManager.tempdir = tmpdir


def draw_theta(theta,eps,upper_bound=1,lower_bound=0,):
    def unif(t,s_lb,s_ub):
        # Draw a random number from either a lower uniform distribution or a higher uniform distribution
        if np.random.rand()>0.5:
            s = np.random.uniform(s_lb,t)
        else :
            s = np.random.uniform(t,s_ub)
        return s

    # Set upper and lower bounds as vectors 
    if type(upper_bound) == int:
        upper_bound = np.ones(len(theta))*upper_bound
    if type(lower_bound) == int:
        lower_bound = np.ones(len(theta))*lower_bound
    
    # Draw thetas 
    theta_proposed = np.zeros(len(theta))
    for i,t in enumerate(theta): 
        lower = max([t-eps[i]/2,lower_bound[i]])
        upper = min([t+eps[i]/2,upper_bound[i]])
        theta_proposed[i] = np.random.uniform(lower,upper,1)
    #if sum(theta_proposed)>1:

    # Scale theta such that the sum is between 0 and 1
    scale_lb = max([sum(theta)-np.mean(eps),0])
    scale_ub = min([sum(theta)+np.mean(eps),1])
    theta_proposed = theta_proposed/sum(theta_proposed)*unif(sum(theta),scale_lb,scale_ub)
    return theta_proposed

def draw_theta_unbound(theta,eps,upper_bound=1,lower_bound=0):
    # Draws a new theta based on previous. 
    # Theta is drawn from a uniform distrubution with width eps and mean theta_i
    if type(upper_bound) == int:
        upper_bound = np.ones(len(theta))*upper_bound
    if type(lower_bound) == int:
        lower_bound = np.ones(len(theta))*lower_bound
    
    theta_proposed = np.zeros(len(theta))
    for i,t in enumerate(theta): 
        lower = max([t-eps[i]/2,lower_bound[i]])
        upper = min([t+eps[i]/2,upper_bound[i]])
        theta_proposed[i] = np.random.uniform(lower,upper,1)

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


def calc_pr(network,cost):
    # Calculate probability estimator, based on solution cost
    cost_0 = network.objective_optimum
    mga_constraint_fullfilment = (cost-cost_0)/cost_0/network.mga_slack
    Pr = lambda c :(-1/(1+np.exp(-100*c+100))+1)
    pr_i = Pr(mga_constraint_fullfilment)
    return pr_i

def calc_pr_narrow_co2(network,base_emission,co2_budget):
    country_emis = get_country_emis(network)
    emis = sum(country_emis.values())
    slack = network.mga_slack
    co2_slack = 0.005
    cost_optimum = network.objective_optimum
    cost = network.objective

    # If emissions are higher than budget + 0.5% then reject
    if emis > co2_budget+(base_emission*co2_slack):
        pr_i = 0
    # If costs are higher than optimum + mga slack
    elif cost > cost_optimum*(1+slack):
        pr_i = 0
    else :
        pr_i = 1
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



def calc_150p_coal_emis(network,emis_factor=1.5):
    # Calculate the alowable emissions, if countries are constrained to not emit more co2 than 
    # the emissions it would take to cover 150% of the country demand with coal power 

    co2_emis_pr_ton = 0.45 # ton emission of co2 pr MWh el produced by coal
    country_loads = network.loads_t.p.groupby(network.buses.country,axis=1).sum()
    country_alowable_emis = country_loads.mul(network.snapshot_weightings,axis=0).sum()*co2_emis_pr_ton*emis_factor

    return country_alowable_emis



def sample(network):

    tmpdir = snakemake.config['solving'].get('tmpdir')
    if tmpdir is not None:
        patch_pyomo_tmpdir(tmpdir)

    mcmc_variables = read_csv(network.mcmc_variables)
    mcmc_variables = [row[0]+row[1] for row in mcmc_variables]

    co2_budget = snakemake.config['co2_budget']
    country_emis = get_country_emis(network)
    try :
        country_emis['EU']
    except :
        country_emis['EU'] = 0

    #if snakemake.config['sampler']['narrow']:
        # Use actual emissions as theta 
    theta = np.array([country_emis[v] for v in mcmc_variables])/co2_budget
    #elif not snakemake.config['sampler']['narrow']:
        # Use previous theta as theta 
    #    theta = str_to_theta(network.theta)
    #else :
    #    raise ValueError('sampler, narrow, not defined in config file')
        
    sigma = np.genfromtxt(snakemake.input[1])
    allowable_emis = calc_150p_coal_emis(network)
    #allowable_emis['EU'] = co2_budget # Allow EU to have all the CO2 budget. This has no real implications 
    allowable_emis['EU'] = 0

    theta_upper_bound = [allowable_emis[key]/co2_budget for key in mcmc_variables] 
    
    #Take step
    #if snakemake.config['sampler']['narrow']:
    theta_proposed = draw_theta_unbound(theta,sigma,lower_bound=0,upper_bound=theta_upper_bound)
    #elif not snakemake.config['sampler']['narrow']:
    #    theta_proposed = draw_theta(theta,sigma,lower_bound=0,upper_bound=theta_upper_bound)
    #else :
    #    raise ValueError('sampler, narrow, not defined in config file')
    
    co2_alocations = co2_budget*theta_proposed
    co2_alocations ={v:t for v,t in zip(mcmc_variables,co2_alocations)}

    snakemake.config['use_local_co2_constraints'] = True
    snakemake.config['local_emission_constraints'] = co2_alocations
    network = solve_network.solve_network(network)
    cost_i = network.objective 

    #if snakemake.config['sampler']['narrow']:
    # Calculate pr_i based on MGA slack 
    #    pr_i = calc_pr(network,cost_i)
    #elif not snakemake.config['sampler']['narrow']:
        # Calculate pr_i based on co2 emissions allone 
    base_emission = snakemake.config['base_emission']
    pr_i = calc_pr_narrow_co2(network,base_emission,co2_budget)
    #else :
    #    raise ValueError('sampler, narrow, not defined in config file')


    return network, pr_i, theta_proposed , theta

def save_network(network,theta,sample,accepted,path):
    network.name = os.path.relpath(os.path.normpath(path))
    network.dual_path = network.name[:-2]+'p'
    pickle.dump((network.duals,network.dualvalues),open(network.dual_path, "wb" ))
    network.sample = sample
    network.accepted = accepted
    network.theta = theta_to_str(theta)
    network.export_to_netcdf(path)
    return network


#%%
if __name__=='__main__':
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        try:
            snakemake = mock_snakemake('run_single_chain',chain=0,sample=1021,run_name='mcmc_2030')
            #os.chdir('..')
        except :
            os.chdir('..')
            snakemake = mock_snakemake('run_single_chain',chain=0,sample=1021,run_name='mcmc_2030')

    import builtins 
    builtins.snakemake = snakemake
    configure_logging(snakemake,skip_handlers=True)

    # Get the sample number of the input network file
    pre,file = os.path.split(snakemake.input[0])
    sample_input = int(re.search('s([0-9]?[0-9]?[0-9]?[0-9])',file).group()[1:])

    # Sample itteratively until the desired sample number is reached
    n_sample = sample_input
    inp = snakemake.input[0]
    out = increment_sample(inp)
    out_prev = inp
    network = pypsa.Network(inp,
                            override_component_attrs=override_component_attrs)
    while n_sample < int(snakemake.wildcards.sample) :
    
        alpha = np.random.rand()
        # Get the likelihood estimate pr_i from the network 
        network, pr_i, theta_proposed, theta_old = sample(network)
        # Accept or reject the sample 
        if alpha<pr_i: # Sample accepted, save solved network
            logging.info('sample accepted')
            network = save_network(network,theta_proposed,n_sample+1,1,out)
        else : # Sample rejected, copy previous network to next
            logging.info('sample rejected')
            # Save rejected network
            folder,file = os.path.split(out)
            rejetc_path = os.path.join(folder,'rejected_'+file)
            network = save_network(network,theta_proposed,n_sample+1,0,rejetc_path)
            # Save previous network
            network = pypsa.Network(out_prev,
                            override_component_attrs=override_component_attrs)
            network.duals , network.dualvalues = pickle.load( open(network.dual_path, "rb" ) )
            network = save_network(network,theta_old,n_sample+1,0,out)

        # Increment file names and sample number
        n_sample = n_sample +1
        out_prev = out
        out = increment_sample(out)
# %%
