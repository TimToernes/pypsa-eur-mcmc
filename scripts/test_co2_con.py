#%%
import pypsa
import numpy as np
from pypsa.linopt import get_var, define_constraints, linexpr, get_dual
#%%

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

# %%
network = pypsa.Network('../data/networks/elec_s_37_lv1.5__Co2L0p25-3H-T-H-B-I-solar+p3-dist1_2030.nc',
#network = pypsa.Network('../data/networks/elec_s_37_lv1.5__Co2L0p25-3H-T-H-B-I-solar+p3-dist1_2030.nc',
#network = pypsa.Network('../data/networks/test.nc',
                        override_component_attrs=override_component_attrs)

#network.global_constraints.loc['CO2Limit','constant'] = np.inf

# %%

nhours = 10
network.set_snapshots(network.snapshots[:nhours])
network.snapshot_weightings[:] = 1#8760./nhours 
# %%
solver = {
  "solver_name": 'gurobi',
  "formulation": 'kirchhoff',
  "pyomo": False,
  "keep_references": False,
  "solver_options": {
    "threads": 4,
    "method": 2, # barrier
    "crossover": 0,
    "BarConvTol": 1.e-6,
    "FeasibilityTol": 1.e-6,
    "AggFill": 0,
    "PreDual": 0,
    }}

stat = network.lopf(**solver,keep_shadowprices=False,)
                    

#%% Set link locations 

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
            network.links.loc[idx,'location'] = country

    # Links connecting to co2 atmosphere without known location are set to belong to EU
    idx_homeless = network.links.query(query_string('co2 atmosphere')).query('location == ""').index
    network.links.loc[idx_homeless,'location'] = 'EU'
    return network

network = set_link_locations(network)

#%%

def get_country_emis(network):

    query_string = lambda x : f'bus0 == "{x}" | bus1 == "{x}" | bus2 == "{x}" | bus3 == "{x}" | bus4 == "{x}"'
    id_co2_links = network.links.query(query_string('co2 atmosphere')).index

    country_codes = network.links.loc[id_co2_links].location.unique()
    country_emis = {code:0 for code in country_codes}

    for country in country_codes:
        idx = network.links.query(f'location == "{country}"').index
        id0 = (network.links.loc[idx] == 'co2 atmosphere')['bus0']
        country_emis[country] -= network.links_t.p0[idx[id0]].sum().sum()
        id1 = (network.links.loc[idx] == 'co2 atmosphere')['bus1']
        country_emis[country] -= network.links_t.p1[idx[id1]].sum().sum()
        id2 = (network.links.loc[idx] == 'co2 atmosphere')['bus2']
        country_emis[country] -= network.links_t.p2[idx[id2]].sum().sum()
        id3 = (network.links.loc[idx] == 'co2 atmosphere')['bus3']
        country_emis[country] -= network.links_t.p3[idx[id3]].sum().sum()
        id4 = (network.links.loc[idx] == 'co2 atmosphere')['bus4']
        country_emis[country] -= network.links_t.p4[idx[id4]].sum().sum()

        if country == 'EU':
            id_load_co2 = network.loads.query('bus == "co2 atmosphere"').index
            co2_load = network.loads.p_set[id_load_co2].sum().sum()*sum(network.snapshot_weightings)
            country_emis[country] -= co2_load

        total_emis = np.sum(list(country_emis.values())) 
    
    return country_emis


country_emis = get_country_emis(network)
total_emis = np.sum(list(country_emis.values()))



#%%

def add_local_co2_constraints(network, snapshots, local_emis):

    efficiency_dict = dict(bus1 = 'efficiency',bus2 = 'efficiency2', bus3 = 'efficiency3' , bus='efficiency4')
    query_string = lambda x : f'bus0 == "{x}" | bus1 == "{x}" | bus2 == "{x}" | bus3 == "{x}" | bus4 == "{x}"'
    id_co2_links = network.links.query(query_string('co2 atmosphere')).index

    country_codes = network.links.loc[id_co2_links].location.unique()

    for country in country_codes:
        idx = network.links.query(f'location == "{country}"').index
        variables = get_var(network,'Link','p').loc[:,idx].values

        efficiencies = []
        for link_id in idx:
            co2_bus = network.links.loc[link_id][network.links.loc[link_id] == 'co2 atmosphere'].index[0]
            if co2_bus == 'bus0':
                efficiency = -1 
            else : 
                efficiency = network.links.loc[link_id,efficiency_dict[co2_bus]]
            efficiencies.append(efficiency)        


        const = np.ones(variables.shape)
        const = (const.T * np.array(network.snapshot_weightings)).T
        const = const * efficiencies

        if country == 'EU':
            try:
                id_load_co2 = network.loads.query('bus == "co2 atmosphere"').index
                co2_load = network.loads_t.p[id_load_co2].sum().sum()
                local_emis[country] += co2_load
            except:
                pass


        expr = linexpr((const,variables)).sum().sum()

        define_constraints(network,expr,'<=',local_emis[country],'national_co2','{}'.format(country))


#%%

def extra_functionality(network,snapshots,local_emis):
    network = set_link_locations(network)
    add_local_co2_constraints(network, snapshots, local_emis)


# %%
query_string = lambda x : f'bus0 == "{x}" | bus1 == "{x}" | bus2 == "{x}" | bus3 == "{x}" | bus4 == "{x}"'
id_co2_links = network.links.query(query_string('co2 atmosphere')).index
country_codes = network.links.loc[id_co2_links].location.unique()

local_emis = {loc:2500 for loc in country_codes}
local_emis['EU'] = 250000

#%%
extra_func = lambda n, s: extra_functionality(n, 
                                            s, 
                                            local_emis
                                             )

stat = network.lopf(**solver,keep_shadowprices=False,
                    extra_functionality=extra_func)

# %%

network.export_to_netcdf('../data/networks/test.nc',)
# %%
