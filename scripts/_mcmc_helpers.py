"""
A collection of function used across the mcmc scripts
"""

import csv
import numpy as np 


def get_theta(network,mcmc_variables):
    theta = []
    co2_emis = calc_co2_emis_pr_node(network)
    for var in mcmc_variables:
        if network.generators.index.isin(var).any():
            theta.append(network.generators.p_nom_opt.loc[network.generators.index.isin(var)].sum())
        elif network.storage_units.index.isin(var).any() :
            theta.append(network.storage_units.p_nom_opt.loc[network.storage_units.index.isin(var)].sum())
        else :
            theta.append(sum([co2_emis[v] for v in var]))

    theta_base = np.genfromtxt(network.theta_base)

    co2_red = (np.array(theta_base) - np.array(theta))/np.array(theta_base)

    return co2_red



def calc_co2_emis_pr_node(network):

    co2_emis = {}
    for bus in network.buses.index:
        local_emis = 0 
        for gen in network.generators.query("bus == '{}' ".format(bus)).index:

            gen_emis = 1/network.generators.efficiency.loc[gen] 
            gen_emis *= (network.snapshot_weightings*network.generators_t.p.T).T.sum().loc[gen]
            try:
                gen_emis *= network.carriers.co2_emissions.loc[network.generators.carrier.loc[gen]]
            except Exception:
                pass
            local_emis += gen_emis
        co2_emis[bus] = local_emis
    return co2_emis


def read_csv(path):
    item = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            item.append(row)
    return item


def write_csv(path,item):
    """ Write a list or numpy array (item) as csv file
    to the specified path """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(item)