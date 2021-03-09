#%%
#from scripts.initialize_network import set_link_locations
import pypsa
import os 
import csv
import numpy as np 
import pandas as pd 
import re 
from _helpers import configure_logging
from solutions import solutions
import multiprocessing as mp
import queue # imported for using queue.Empty exception
from itertools import product
from functools import partial
import time 
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from _mcmc_helpers import *
os.chdir('..')


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


# %% import datasets 

run_name = 'h100_model'
#run_name = 'h99_model'
network = pypsa.Network(f'results/{run_name}/network_c0_s1.nc',override_component_attrs=override_component_attrs)
mcmc_variables = read_csv(network.mcmc_variables)
mcmc_variables = [row[0]+row[1] for row in mcmc_variables]
df_secondary = pd.read_csv(f'results/{run_name}/result_secondary_metrics.csv',index_col=0)
df_sum = pd.read_csv(f'results/{run_name}/result_sum_vars.csv',index_col=0)
df_gen_p = pd.read_csv(f'results/{run_name}/result_gen_p.csv',index_col=0)
df_co2 = pd.read_csv(f'results/{run_name}/result_co2_pr_node.csv',index_col=0)
df_chain = pd.read_csv(f'results/{run_name}/result_chain.csv',index_col=0)
df_links = pd.read_csv(f'results/{run_name}/result_links.csv',index_col=0)
df_co2 = pd.read_csv(f'results/{run_name}/result_co2_pr_node.csv',index_col=0)
df_store = pd.read_csv(f'results/{run_name}/result_store_p.csv',index_col=0)

theta = pd.read_csv(f'results/{run_name}/theta.csv',index_col=0)
theta.columns = [x for x in mcmc_variables+['s','c','a']]
theta['id'] = theta.index

emis_1990 = 1481895952.8

df_pop = pd.read_csv('data/API_SP.POP.TOTL_DS2_en_csv_v2_2106202.csv',sep=',',index_col=0,skiprows=3)
df_gdp = pd.read_csv('data/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_2055594.csv',sep=',',index_col=0,skiprows=3)


#%%#########################################
####### Data postprocessing ################
############################################

opt_name = {"Store": "e", "Line" : "s", "Transformer" : "s"}
label = 'test'
nodal_costs = pd.Series()

def calculate_nodal_costs(n,nodal_costs):
    #Beware this also has extraneous locations for country (e.g. biomass) or continent-wide (e.g. fossil gas/oil) stuff
    for c in n.iterate_components(n.branch_components|n.controllable_one_port_components^{"Load"}):
        c.df["capital_costs"] = c.df.capital_cost*c.df[opt_name.get(c.name,"p") + "_nom_opt"]
        capital_costs = c.df.groupby(["location","carrier"])["capital_costs"].sum()
        index = pd.MultiIndex.from_tuples([(c.list_name,"capital") + t for t in capital_costs.index.to_list()])
        nodal_costs = nodal_costs.reindex(index|nodal_costs.index)
        nodal_costs.loc[index] = capital_costs.values

        if c.name == "Link":
            p = c.pnl.p0.multiply(n.snapshot_weightings,axis=0).sum()
        elif c.name == "Line":
            continue
        elif c.name == "StorageUnit":
            p_all = c.pnl.p.multiply(n.snapshot_weightings,axis=0)
            p_all[p_all < 0.] = 0.
            p = p_all.sum()
        else:
            p = c.pnl.p.multiply(n.snapshot_weightings,axis=0).sum()

        #correct sequestration cost
        if c.name == "Store":
            items = c.df.index[(c.df.carrier == "co2 stored") & (c.df.marginal_cost <= -100.)]
            c.df.loc[items,"marginal_cost"] = -20.

        c.df["marginal_costs"] = p*c.df.marginal_cost
        marginal_costs = c.df.groupby(["location","carrier"])["marginal_costs"].sum()
        index = pd.MultiIndex.from_tuples([(c.list_name,"marginal") + t for t in marginal_costs.index.to_list()])
        nodal_costs = nodal_costs.reindex(index|nodal_costs.index)
        nodal_costs.loc[index] = marginal_costs.values

    return nodal_costs


def assign_locations(n):
    for c in n.iterate_components(n.one_port_components|n.branch_components):

        ifind = pd.Series(c.df.index.str.find(" ",start=4),c.df.index)

        for i in ifind.unique():
            names = ifind.index[ifind == i]

            if i == -1:
                c.df.loc[names,'location'] = ""
            else:
                c.df.loc[names,'location'] = names.str[:i]


assign_locations(network)
nodal_costs = calculate_nodal_costs(network,nodal_costs)


#%%

def set_link_locataions(network):
    network.links['location'] = ""
    network.generators['location'] = ""
    network.lines['location'] = ""
    network.stores['location'] = ""
    network.storage_units['location']

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
            idx = network.links.query(query_string(bus))['location'].index
            network.links.loc[idx,'location'] = country

            idx = network.generators.query(f"bus == '{bus}'")['location'].index
            network.generators.loc[idx,'location'] = country

    # Links connecting to co2 atmosphere without known location are set to belong to EU
    idx_homeless = network.links.query(query_string('co2 atmosphere')).query('location == ""').index
    network.links.loc[idx_homeless,'location'] = 'EU'
    return network

# Calc data for cost increase and co2 reduction 

cost_increase = (df_secondary.system_cost-network.objective_optimum)/network.objective_optimum*100

base_emis = emis_1990

df_secondary['cost_increase'] = cost_increase
df_secondary['co2_reduction'] = 100 - df_co2.sum(axis=1)/base_emis * 100


# Dataset with with aggregated technology capacities 
network = set_link_locataions(network)
idx = network.links.query('location != "EU" & location != ""').index
df_link_sum = df_links[idx].groupby(network.links.carrier,axis=1).sum()

df_gen_sum = df_gen_p.groupby(network.generators.carrier,axis=1).sum()
#df_gen_sum.pop('oil')

df_tech_sum = pd.concat([df_link_sum,df_gen_sum],axis=1)
df_tech_sum['wind'] = df_tech_sum[['offwind','offwind-ac','offwind-dc','onwind']].sum(axis=1)


###########################################
############### filters ###################
###########################################
# create filter for 150 CO2 reduction compared to coal production 

country_loads = network.loads_t.p.sum().groupby(network.buses.country).sum()
co2_emis_pr_ton = 0.095
alowable_emis_pr_country = country_loads * co2_emis_pr_ton *1.5 * network.snapshot_weightings[0]
df_co2_contry = df_co2.groupby(network.buses.country,axis=1).sum()
df_co2_contry = df_co2_contry.loc[:,df_co2_contry.columns != '']
filt_co2_150p = (df_co2_contry<alowable_emis_pr_country).all(axis=1)

# create a filter for minimum backup capacity

df_ocgt = df_gen_p.loc[:,network.generators.carrier=='OCGT']
ocgt_pr_country = df_ocgt.groupby(network.generators.bus,axis=1).sum().groupby(network.buses.country,axis=1).sum()
country_max_load = network.loads_t.p.max().groupby(network.buses.country).sum()
filt_backup = (ocgt_pr_country>0.5*country_max_load).all(axis=1)

# filter for 10% cost increase 
filt_cost = df_secondary['cost_increase'] < 10


#%%##########################################
# ############ plots ########################
# ###########################################
# Corrolelogram cost vs co2 

df = df_secondary[['cost_increase']]
df['co2_reduction'] = 100 - df_co2.sum(axis=1)/base_emis * 100

df['filt_cost'] = filt_cost

sns_plot = sns.pairplot(df, kind="hist", diag_kind='hist',hue='filt_cost')
#plt.suptitle('Scenarios with less than 150% local emisons compared to 100% coal production')
#plt.suptitle('Scenarios where all countries have more than 10% fosil fuel backup')

sns_plot.savefig(f'graphics/cost_co2_{run_name}.jpeg')
fig = sns_plot.fig
fig.show()

#%% Corrolelogram tech

#df = df_sum[['OCGT', 'offwind-ac', 'offwind-dc', 'onwind', 'solar', 'transmission', 'H2', 'battery',]]
df = df_tech_sum[['OCGT','H2 Electrolysis','SMR','solar','wind','gas']]
#df = df[df_chain.s>200]
#df['co2<150p'] = filt_co2_150p
#df['10p_backup'] = filt_backup
#df['c'] = df_chain.c
#df = df[df_secondary.co2_reduction>50]
#df['c'] = theta.c
df['filt_cost'] = filt_cost

 
# without regression
sns_plot = sns.pairplot(df, kind="hist", diag_kind='hist',hue='filt_cost')
#plt.suptitle('Scenarios with less than 150% local emisons compared to 100% coal production')
#plt.suptitle('Scenarios where all countries have more than 10% fosil fuel backup')

#sns_plot.savefig(f'graphics/tech_{run_name}.jpeg')
fig = sns_plot.fig
fig.show()

#%% Corrolelogram secondary metrics


df_secondary['transmission'] = df_sum['transmission']
#df = sns.load_dataset('iris')
df = df_secondary[['cost_increase','gini_co2','gini',]]
df['co2_reduction'] = df['co2_reduction'] = 100 - df_co2.sum(axis=1)/base_emis * 100
#df = df[df_secondary.cost_increase<7]
#df = df[df_secondary.co2_reduction>50]
#df['co2<150p'] = filt_co2_150p
#df['10p_backup'] = filt_backup
#df['c'] = theta.c
df['filt_cost'] = filt_cost



sns_plot = sns.pairplot(df, kind="hist", diag_kind='hist',hue='filt_cost')
#plt.suptitle('Scenarios with less than 150% emisons compared to 100% coal production')
#plt.suptitle('Scenarios where all countries have more than 10% fosil fuel backup')

#sns_plot.savefig(f'graphics/secondary_{run_name}.jpeg')
sns_plot.fig.show()

#%% Plot of chain development over time 

accept_percent = sum(theta.a)/theta.shape[0]*100
print(f'Acceptance {accept_percent:.1f}%')

theta_long = pd.wide_to_long(theta,stubnames=['theta_'],i='id',j='theta')
theta_long = theta_long.reset_index(level=['theta'])

sns.set_theme(style="ticks")
# Define the palette as a list to specify exact values
palette = sns.color_palette("rocket", as_cmap=True)

# Plot the lines on two facets
sns.relplot(
    data=theta_long,
    x="s", y="theta_",
    hue="theta",
    palette=palette,
    col='c',
    kind="line",
    height=5, aspect=.75,)

#%% plot of co2 emis 

df = df_co2[['DE','DK','FR','ES','PL']]

df['filt_cost'] = filt_cost


sns_plot = sns.pairplot(df, kind="hist", diag_kind='hist',hue='filt_cost')
plt.suptitle('CO2 emissions pr country [T CO2]')

#sns_plot.map_lower(sns.regplot)
#sns_plot.savefig('test2.pdf')

sns_plot.savefig(f'graphics/co2_emissions_{run_name}.jpeg')
sns_plot.fig.show()

#%% plot of thetas 

df = theta[['DK','DE','FR','PL']]

sns_plot = sns.pairplot(df, kind="hist", diag_kind='hist')#,hue='10p_backup')
plt.suptitle('CO2 emissions pr country [T CO2]')

#sns_plot.map_lower(sns.regplot)
#sns_plot.savefig('test2.pdf')

sns_plot.savefig(f'graphics/co2_emissions_{run_name}.jpeg')
sns_plot.fig.show()


#%%

def calc_co2_gini(network):

    co2_emis = calc_co2_emis_pr_node(network)
    co2_emis = pd.Series(co2_emis)

    #bus_total_prod = network.generators_t.p.sum().groupby(network.generators.bus).sum()
    load_total= network.loads_t.p_set.sum()

    rel_demand = load_total/sum(load_total)
    rel_generation = co2_emis/sum(co2_emis)

    # Rearange demand and generation to be of increasing magnitude
    idy = np.argsort(rel_generation/rel_demand)
    rel_demand = rel_demand[idy]
    rel_generation = rel_generation[idy]

    # Calculate cumulative sum and add [0,0 as point
    rel_demand = np.cumsum(rel_demand)
    rel_demand = np.concatenate([[0],rel_demand])
    rel_generation = np.cumsum(rel_generation)
    rel_generation = np.concatenate([[0],rel_generation])

    lorenz_integral= 0
    for i in range(len(rel_demand)-1):
        lorenz_integral += (rel_demand[i+1]-rel_demand[i])*(rel_generation[i+1]-rel_generation[i])/2 + (rel_demand[i+1]-rel_demand[i])*rel_generation[i]

    gini = 1- 2*lorenz_integral
    return gini


#%%

from iso3166 import countries
import plotly.graph_objects as go 
import pypsa
from _mcmc_helpers import calc_co2_emis_pr_node, read_csv

#network = pypsa.Network(f'inter_results/{run_name}/network_c0_s1.nc')
mcmc_variables = read_csv(f'results/{run_name}/mcmc_variables.csv')

df_co2['s'] = df_chain['s']
df_co2['c'] = df_chain['c']
df_co2['co2_reduction'] = df_secondary.co2_reduction

theta_i = df_co2.query('s==151 & c==6')
base_emis = 38750000

co2_emis = {}
for mcmc_var in mcmc_variables:
    for bus in mcmc_var:
        alpha2 = bus[:2]
        alpha3 = countries.get(alpha2).alpha3
        if alpha3 in co2_emis.keys():
            co2_emis[alpha3] += float(theta_i[bus])
        else :
            co2_emis[alpha3] = float(theta_i[bus])



#co2_emis2 = calc_co2_emis_pr_node(network)

fig = go.Figure()


fig.add_trace(go.Choropleth(
                    geo='geo1',
                    locations = list(co2_emis.keys()),
                    z = list(co2_emis.values()),#/area,
                    text = list(co2_emis.keys()),
                    colorscale = 'Thermal',
                    autocolorscale=False,
                    #zmax=283444,
                    zmin=0,
                    reversescale=False,
                    marker_line_color='darkgray',
                    marker_line_width=0.5,
                    #colorbar_tickprefix = '',
                    #colorbar_title = 'Potential [MWh/km^2]'
                        )) 

fig.update_geos(
        scope = 'europe',
        projection_type = 'azimuthal equal area',
        showland = True,
        landcolor = 'rgb(243, 243, 243)',
        countrycolor = 'rgb(204, 204, 204)',
        lataxis = dict(
            range = [35, 64],
            showgrid = False
        ),
        lonaxis = dict(
            range = [-11, 26],
            showgrid = False
        )
    ),

fig.update_layout(
    autosize=False,
    width=900,
    height=500,
    showlegend=False,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)


fig.show()
# %%
