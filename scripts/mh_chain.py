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
        if np.random.rand()>0.5:
            s = np.random.uniform(s_lb,t)
        else :
            s = np.random.uniform(t,s_ub)
        return s

    if type(upper_bound) == int:
        upper_bound = np.ones(len(theta))*upper_bound
    if type(lower_bound) == int:
        lower_bound = np.ones(len(theta))*lower_bound
    

    theta_proposed = np.zeros(len(theta))
    for i,t in enumerate(theta): 
        lower = max([t-eps[i]/2,lower_bound[i]])
        upper = min([t+eps[i]/2,upper_bound[i]])
        theta_proposed[i] = np.random.uniform(lower,upper,1)
    #if sum(theta_proposed)>1:
    scale_lb = max([sum(theta)-np.mean(eps),0])
    scale_ub = min([sum(theta)+np.mean(eps),1])
    theta_proposed = theta_proposed/sum(theta_proposed)*unif(sum(theta),scale_lb,scale_ub)
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


#def solve_network(network,variables,theta,tmpdir):
#    extra_func = lambda n, s: extra_functionality(n,
#                                                s,
#                                                variables,
#                                                theta)
#
#    stat = network.lopf(**snakemake.config.get('solver'),
#                        solver_dir = tmpdir,
#                        extra_functionality=extra_func)
#    logging.info(stat)
#
#    if str(stat[1])=='infeasible or unbounded':
#        network.objective = np.inf
#    return network


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



#def get_country_emis(network):
#
#    query_string = lambda x : f'bus0 == "{x}" | bus1 == "{x}" | bus2 == "{x}" | bus3 == "{x}" | bus4 == "{x}"'
#    id_co2_links = network.links.query(query_string('co2 atmosphere')).index
#
#    country_codes = network.links.loc[id_co2_links].location.unique()
#    country_emis = {code:0 for code in country_codes}
#
#    for country in country_codes:
#        idx = network.links.query(f'location == "{country}"').index
#        id0 = (network.links.loc[idx] == 'co2 atmosphere')['bus0']
#        country_emis[country] -= network.links_t.p0[idx[id0]].sum(axis=1).mul(network.snapshot_weightings).sum()
#        id1 = (network.links.loc[idx] == 'co2 atmosphere')['bus1']
#        country_emis[country] -= network.links_t.p1[idx[id1]].sum(axis=1).mul(network.snapshot_weightings).sum()
#        id2 = (network.links.loc[idx] == 'co2 atmosphere')['bus2']
#        country_emis[country] -= network.links_t.p2[idx[id2]].sum(axis=1).mul(network.snapshot_weightings).sum()
#        id3 = (network.links.loc[idx] == 'co2 atmosphere')['bus3']
#        country_emis[country] -= network.links_t.p3[idx[id3]].sum(axis=1).mul(network.snapshot_weightings).sum()
#        id4 = (network.links.loc[idx] == 'co2 atmosphere')['bus4']
#        country_emis[country] -= network.links_t.p4[idx[id4]].sum(axis=1).mul(network.snapshot_weightings).sum()
#
#        if country == 'EU':
#            id_load_co2 = network.loads.query('bus == "co2 atmosphere"').index
#            co2_load = network.loads.p_set[id_load_co2].sum().sum()*sum(network.snapshot_weightings)
#            country_emis[country] -= co2_load
#
#        total_emis = np.sum(list(country_emis.values())) 
#    
#    return country_emis


def calc_150p_coal_emis(network,emis_factor=1.5):
    # Calculate the alowable emissions, if countries are constrained to not emit more co2 than 
    # the emissions it would take to cover 150% of the country demand with coal power 

    co2_emis_pr_ton = 0.095 # ton emission of co2 pr MWh el produced by coal
    country_loads = network.loads_t.p.groupby(network.buses.country,axis=1).sum()
    country_alowable_emis = country_loads.mul(network.snapshot_weightings,axis=0).sum()*co2_emis_pr_ton*emis_factor

    return country_alowable_emis



def sample(network):

    tmpdir = snakemake.config['tmpdir']
    if tmpdir is not None:
        patch_pyomo_tmpdir(tmpdir)

    mcmc_variables = read_csv(network.mcmc_variables)
    mcmc_variables = [row[0]+row[1] for row in mcmc_variables]

    co2_budget = snakemake.config['co2_budget']
    country_emis = get_country_emis(network)
    theta = np.array([country_emis[v] for v in mcmc_variables])/co2_budget
    #theta = get_theta(network,mcmc_variables,snakemake.config['co2_budget'])
    #sigma = np.array(read_csv(snakemake.input[1])).astype(float)
    sigma = np.genfromtxt(snakemake.input[1])
    allowable_emis = calc_150p_coal_emis(network)
    allowable_emis['EU'] = co2_budget # Allow EU to have all the CO2 budget. This has no real implications 

    theta_upper_bound = [allowable_emis[key]/co2_budget for key in mcmc_variables] 



    # Evaluate capital costs of the proposed solution (theta_porposed) and use this as early rejection criteria
#    cost_early = calc_capital_cost(network,mcmc_variables,theta_proposed)
#    pr_early = calc_pr(network,cost_early)

#    if pr_early>alpha: # Sample not rejected based on early evaluation
        # Evaluate network at new point 
    #theta_base = np.genfromtxt(network.theta_base)
    
    #Take step
    theta_proposed = draw_theta(theta,sigma,lower_bound=0,upper_bound=theta_upper_bound)

    co2_alocations = co2_budget*theta_proposed
    co2_alocations ={v:t for v,t in zip(mcmc_variables,co2_alocations)}

    if not all([allowable_emis[key]>=co2_alocations[key]-1e-3 for key in allowable_emis.keys()]):
        print('theta does not satisfy 150% co2 constraint')

    snakemake.config['use_local_co2_constraints'] = True
    snakemake.config['local_emission_constraints'] = co2_alocations

    network = solve_network.prepare_network(network)
    network = solve_network.solve_network(network)
    #network = solve_network(network,mcmc_variables,co2_alocations,tmpdir)
    cost_i = network.objective 
    pr_i = calc_pr(network,cost_i)
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

    import builtins 
    builtins.snakemake = snakemake
    configure_logging(snakemake,skip_handlers=True)

    # Get the sample number of the input network file
    pre,file = os.path.split(snakemake.input[0])
    sample_input = int(re.search('s([0-9]?[0-9]?[0-9])',file).group()[1:])

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
        network_old = network.copy()
        network, pr_i = sample(network)
        # Accept or reject the sample 
        if alpha<pr_i: # Sample accepted, save solved network
            logging.info('sample accepted')
            network.sample = n_sample+1
            network.accepted = 1
            network.export_to_netcdf(out)
        else : # Sample rejected, copy previous network to next
            logging.info('sample rejected')
            network = pypsa.Network(out_prev,
                            override_component_attrs=override_component_attrs)
            network.sample = n_sample+1
            network.accepted = 0 
            network.export_to_netcdf(out)
            #shutil.copyfile(inp,out)

        # Increment file names and sample number
        n_sample = n_sample +1
        out_prev = out
        out = increment_sample(out)


# %%

#theta = np.array([1,-0.4])

#sigma = np.array([[0.5,0],[0,0.5]])

# %%
