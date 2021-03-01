"""
A collection of function used across the mcmc scripts
"""

import csv
import numpy as np 


def get_theta(network,mcmc_variables,co2_budget):
    theta = []
    co2_emis = calc_co2_emis_pr_node(network)
    for var in mcmc_variables:
        if network.generators.index.isin(var).any():
            theta.append(network.generators.p_nom_opt.loc[network.generators.index.isin(var)].sum())
        elif network.storage_units.index.isin(var).any() :
            theta.append(network.storage_units.p_nom_opt.loc[network.storage_units.index.isin(var)].sum())
        else :
            theta.append(sum([co2_emis[v] for v in var]))

    co2_red = (np.array(theta))/co2_budget

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



def get_country_emis(network):

    query_string = lambda x : f'bus0 == "{x}" | bus1 == "{x}" | bus2 == "{x}" | bus3 == "{x}" | bus4 == "{x}"'
    id_co2_links = network.links.query(query_string('co2 atmosphere')).index

    country_codes = network.links.loc[id_co2_links].location.unique()
    country_emis = {code:0 for code in country_codes}

    for country in country_codes:
        idx = network.links.query(f'location == "{country}"').index
        id0 = (network.links.loc[idx] == 'co2 atmosphere')['bus0']
        country_emis[country] -= network.links_t.p0[idx[id0]].sum(axis=1).mul(network.snapshot_weightings).sum()
        id1 = (network.links.loc[idx] == 'co2 atmosphere')['bus1']
        country_emis[country] -= network.links_t.p1[idx[id1]].sum(axis=1).mul(network.snapshot_weightings).sum()
        id2 = (network.links.loc[idx] == 'co2 atmosphere')['bus2']
        country_emis[country] -= network.links_t.p2[idx[id2]].sum(axis=1).mul(network.snapshot_weightings).sum()
        id3 = (network.links.loc[idx] == 'co2 atmosphere')['bus3']
        country_emis[country] -= network.links_t.p3[idx[id3]].sum(axis=1).mul(network.snapshot_weightings).sum()
        id4 = (network.links.loc[idx] == 'co2 atmosphere')['bus4']
        country_emis[country] -= network.links_t.p4[idx[id4]].sum(axis=1).mul(network.snapshot_weightings).sum()

        if country == 'EU':
            id_load_co2 = network.loads.query('bus == "co2 atmosphere"').index
            co2_load = network.loads.p_set[id_load_co2].sum().sum()*sum(network.snapshot_weightings)
            country_emis[country] -= co2_load

        total_emis = np.sum(list(country_emis.values())) 
    
    return country_emis