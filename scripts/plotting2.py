#%%
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
#os.chdir('..')
# %% import datasets 

run_name = 'h8759_model'
#run_name = 'h99_model'

#network = pypsa.Network('inter_results/network_c0_s10.nc')
network = pypsa.Network(f'results/{run_name}/network_c0_s1.nc')
df_secondary = pd.read_csv(f'results/{run_name}/result_secondary_metrics.csv',index_col=0)
df_sum = pd.read_csv(f'results/{run_name}/result_sum_vars.csv',index_col=0)
df_gen_p = pd.read_csv(f'results/{run_name}/result_gen_p.csv',index_col=0)
df_co2 = pd.read_csv(f'results/{run_name}/result_co2_pr_node.csv',index_col=0)
df_chain = pd.read_csv(f'results/{run_name}/result_chain.csv',index_col=0)


theta = pd.read_csv(f'results/{run_name}/theta.csv',index_col=0)
theta.columns = [f'theta_{x}' for x in range(33)]+['s','c','a']
theta['id'] = theta.index
# %% Calc data for cost increase and co2 reduction 

cost_increase = (df_secondary.system_cost-network.objective_optimum)/network.objective_optimum*100

base_emis = 38750000
co2_red = (base_emis - df_secondary.loc[:,'co2_emission'])/base_emis*100

df_secondary['cost_increase'] = cost_increase
df_secondary['co2_reduction'] = co2_red

#%% create filter for 150 CO2 reduction compared to coal production 

country_loads = network.loads_t.p.sum().groupby(network.buses.country).sum()
co2_emis_pr_ton = 0.095
alowable_emis_pr_country = country_loads * co2_emis_pr_ton *1.5
df_co2_contry = df_co2.groupby(network.buses.country,axis=1).sum()
filt_co2_150p = (df_co2_contry<alowable_emis_pr_country).all(axis=1)

#%%

df_ocgt = df_gen_p.loc[:,network.generators.carrier=='OCGT']

ocgt_pr_country = df_ocgt.groupby(network.generators.bus,axis=1).sum().groupby(network.buses.country,axis=1).sum()

country_max_load = network.loads_t.p.max().groupby(network.buses.country).sum()

filt_backup = (ocgt_pr_country>0.1*country_max_load).all(axis=1)

#%% Corrolelogram tech

df = df_sum[['OCGT', 'offwind-ac', 'offwind-dc', 'onwind', 'solar', 'transmission', 'H2', 'battery',]]
#df = df[df_chain.s>200]
df['co2<150p'] = filt_co2_150p
#df['10p_backup'] = filt_backup
#df['c'] = df_chain.c
#df = df[df_secondary.co2_reduction>50]
#df['c'] = theta.c
# with regression
#sns.pairplot(df, kind="reg")
#plt.show()
 
# without regression
sns_plot = sns.pairplot(df, kind="hist", diag_kind='kde',hue='co2<150p')
plt.suptitle('Scenarios with less than 150% local emisons compared to 100% coal production')
#plt.suptitle('Scenarios where all countries have more than 10% fosil fuel backup')

#sns_plot.savefig(f'graphics/tech_{run_name}_150p_co2_cap.jpeg')
fig = sns_plot.fig
fig.show()

#%% Corrolelogram secondary metrics

import matplotlib.pyplot as plt
import seaborn as sns

df_secondary['transmission'] = df_sum['transmission']
#df = sns.load_dataset('iris')
df = df_secondary[['cost_increase','co2_reduction','autoarky','gini_co2','gini','transmission']]
#df = df[df_secondary.cost_increase<7]
#df = df[df_secondary.co2_reduction>50]
#df['co2<150p'] = filt_co2_150p
df['10p_backup'] = filt_backup
#df['c'] = theta.c
# with regression
#sns.pairplot(df, kind="reg")
#plt.show()
# without regression
sns_plot = sns.pairplot(df, kind="hist", diag_kind='kde',hue='10p_backup')
#plt.suptitle('Scenarios with less than 150% emisons compared to 100% coal production')
plt.suptitle('Scenarios where all countries have more than 10% fosil fuel backup')

#sns_plot.savefig(f'graphics/secondary_{run_name}_10p_backup.jpeg')
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

#%%

df_co2['DK0 1'] = df_co2[['DK0 0','DK3 0']].sum(axis=1)
df_co2['GB0 0'] = df_co2[['GB4 0','GB5 0']].sum(axis=1)

df_co2['cost_increase'] = df_secondary['cost_increase']

df = df_co2[['DK0 1','DE0 0','FR0 0','GB0 0','cost_increase']]
#df = df[df_secondary.cost_increase<25]
#df = df[df_secondary.co2_reduction>80]
df['10p_backup'] = filt_backup
#df['co2<150p'] = filt_co2_150p

#df['c'] = theta.c
# with regression
#sns.pairplot(df, kind="reg")
#plt.show()
 
# without regression
sns_plot = sns.pairplot(df, kind="hist", diag_kind='kde',hue='10p_backup')
#sns_plot.map_lower(sns.regplot)
#sns_plot.savefig('test2.pdf')

sns_plot.savefig(f'graphics/theta_{run_name}_10p_backup.jpeg')
sns_plot.fig.show()


#%%
from _mcmc_helpers import calc_co2_emis_pr_node

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
