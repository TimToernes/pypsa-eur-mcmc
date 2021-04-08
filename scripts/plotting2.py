#%%

import pypsa
import os 
import numpy as np 
import pandas as pd 
import multiprocessing as mp
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.interactive(False)
import seaborn as sns
from _mcmc_helpers import *



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


try :
    network = pypsa.Network('data/networks/elec_s_37_lv1.5__Co2L0p50-3H-solar+p3-dist1_2030.nc',
                            override_component_attrs=override_component_attrs)
except : 
    os.chdir('..')
    network = pypsa.Network('data/networks/elec_s_37_lv1.5__Co2L0p50-3H-solar+p3-dist1_2030.nc',
                            override_component_attrs=override_component_attrs)

years = [2030,2040,2050]

load_sweep = True

# El 1510 MT
# Heat 723 MT
# Transport 784 MT


emis_1990 =   1481895952.8
emis_1990_H = 2206933437.8055553

base_emis = emis_1990


df_names = dict(df_sum='result_sum_vars.csv',
                df_secondary='result_secondary_metrics.csv',
                df_gen_p='result_gen_p.csv',
                df_gen_e='result_gen_E.csv',
                df_co2='result_co2_pr_node.csv',
                df_chain='result_df_chain.csv',
                df_links='result_links.csv',
                df_lines='result_lines.csv',
                df_store_E='result_store_E.csv',
                df_store_P='result_store_P.csv',
                df_storage_E='result_storeage_unit_E.csv',
                df_storage_P='result_storeage_unit_P.csv',
                df_nodal_cost='result_nodal_costs.csv',
                df_theta='result_theta.csv'
                )

df_sum=pd.DataFrame()
df_secondary=pd.DataFrame()
df_gen_p=pd.DataFrame()
df_gen_e=pd.DataFrame()
df_co2=pd.DataFrame()
df_chain=pd.DataFrame()
df_links=pd.DataFrame()
df_lines=pd.DataFrame()
df_store_E=pd.DataFrame()
df_store_P=pd.DataFrame()
df_storage_E=pd.DataFrame()
df_storage_P=pd.DataFrame()
df_nodal_cost=pd.DataFrame()
df_theta=pd.DataFrame()
df_secondary_sweep = pd.DataFrame()
df_store_P_sweep = pd.DataFrame()
df_co2_sweep = pd.DataFrame()


dfs = {}
networks = {}

for year in years:
    run_name = f'mcmc_{year}'
    #run_name = 'h99_model'
    networks[year] = pypsa.Network(f'results/{run_name}/network_c0_s1.nc',override_component_attrs=override_component_attrs)

    for df_name in df_names.keys():
        if df_name =='df_nodal_cost':
            df = pd.read_csv(f'results/{run_name}/'+df_names[df_name],index_col=0,header=[0,1,2,3])
        else : 
            df = pd.read_csv(f'results/{run_name}/'+df_names[df_name],index_col=0)
        df['year'] = year
        try :
            dfs[df_name] = pd.concat((dfs[df_name],df),ignore_index=True)
            vars()[df_name] = dfs[df_name]
        except Exception:
            dfs[df_name] = df
            vars()[df_name] = dfs[df_name]

network = networks[years[0]]
mcmc_variables = read_csv(networks[year].mcmc_variables)
mcmc_variables = [row[0]+row[1] for row in mcmc_variables]
                
df_pop = pd.read_csv('data/API_SP.POP.TOTL_DS2_en_csv_v2_2106202.csv',sep=',',index_col=0,skiprows=3)
df_gdp = pd.read_csv('data/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_2055594.csv',sep=',',index_col=0,skiprows=3)

if load_sweep :
    for year in years:
        run_name = f'sweep_e_{year}'
        #run_name = 'h99_model'
        #networks[year] = pypsa.Network(f'results/{run_name}/network_c0_s1.nc',override_component_attrs=override_component_attrs)

        for df_name in df_names.keys():
            if df_name =='df_nodal_cost':
                df = pd.read_csv(f'results/{run_name}/'+df_names[df_name],index_col=0,header=[0,1,2,3])
            else : 
                df = pd.read_csv(f'results/{run_name}/'+df_names[df_name],index_col=0)
            df['year'] = year
            df_name = df_name + '_sweep'
            try :
                dfs[df_name] = pd.concat((dfs[df_name],df),ignore_index=True)
                vars()[df_name] = dfs[df_name]
            except : 
                dfs[df_name] = df
                vars()[df_name] = dfs[df_name]

#%%#########################################
####### Data postprocessing ################
############################################


def set_link_locataions(network):
    network.links['location'] = ""
    network.generators['location'] = ""
    network.lines['location'] = ""
    network.stores['location'] = ""
    #network.storage_units['location']

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

df_secondary['cost_increase'] = cost_increase
df_secondary['co2_reduction'] = 100 - df_co2.sum(axis=1)/base_emis * 100


# Dataset with with aggregated technology capacities
network = set_link_locataions(network)
idx = network.links.query('location != "EU" & location != ""').index
df_link_sum = df_links[idx].groupby(network.links.carrier,axis=1).sum()

stores_carrier = pd.concat((networks[key].stores.carrier for key in networks))
stores_carrier = stores_carrier[~stores_carrier.index.duplicated()]
df_store_sum = df_store_P.groupby(stores_carrier,axis=1).sum()
df_store_sum.columns = [c + '_store' for c in df_store_sum.columns]
#df_gen_sum = df_gen_p.groupby(network.generators.carrier,axis=1).sum()
df_gen_sum = df_sum
#df_gen_sum.pop('oil')

df_tech_sum = pd.concat([df_link_sum,df_gen_sum,df_store_sum],axis=1)
df_tech_sum['wind'] = df_tech_sum[['offwind','offwind-ac','offwind-dc','onwind']].sum(axis=1)

# Dataset with aggregated technology energy production 
df_tech_e_sum = df_gen_e.groupby(network.generators.carrier,axis=1).sum()


#%%##########################################
# ############ plots ########################
# ###########################################
# ####### Corrolelogram cost vs co2 #########

def plot_cost_vs_co2(prefix='',save=False):

    co2_emis_levels = {2030:1-0.45,
                        2040:1-0.225,
                        2050:1-0.05}

    run_name = 'elec'
    mga_slack = 0.2

    df = df_secondary[['system_cost']]
    #df['co2 emission'] = df_co2.sum(axis=1)
    df['co2 emission'] =df_store_P['co2 atmosphere']
    df['co2 reduction'] = 1-(df['co2 emission']/base_emis )
    #df['co2 reduction'] = (1-df_co2.sum(axis=1)/base_emis)*100
    df['year'] = df_co2['year']

    cost_limits = [df['system_cost'].min(),df['system_cost'].max()]
    co2_limits = [df['co2 reduction'].min(),df['co2 reduction'].max()]


    df_optimal = df_secondary_sweep[['system_cost']]
    df_optimal['co2 emission'] = df_store_P_sweep['co2 atmosphere']
    df_optimal['co2 reduction'] = 1-(df_optimal['co2 emission']/base_emis )
    df_optimal['year'] = df_co2_sweep['year']


    def plot_optimum(xdata,ydata,**kwargs):
        plt.gca()
        sns.lineplot(data=df_optimal,
                            x='co2 reduction',
                            y='system_cost',
                            hue='year',
                            palette='Set1'
                            )


    # Function for plotting vertical lines for co2 limit 
    def plot_lower(xdata,ydata,**kwargs):
        plt.gca().vlines(co2_emis_levels[kwargs['label']],
                        ymin=cost_limits[0],
                        ymax=cost_limits[1],
                        colors=kwargs['color'])
        #cost = networks[kwargs['label']].objective_optimum
        #plt.gca().hlines([cost],
        #                 xmin=co2_limits[0],
        #                 xmax=co2_limits[1],
        #                 colors=kwargs['color'])


    #df['filt_cost'] = filt_cost

    sns_plot = sns.pairplot(df, 
                            vars=['co2 reduction','system_cost'],
                            kind="hist", 
                            diag_kind='hist',
                            hue='year',
                            plot_kws=dict(bins=60),
                            diag_kws=dict(bins=60,kde=False,log_scale=False),
                            aspect=1.6,
                            height=3,
                            palette='Set1')
    #plt.suptitle('Scenarios with less than 150% local emisons compared to 100% coal production')
    #plt.suptitle('Scenarios where all countries have more than 10% fosil fuel backup')

    cost_limits = [df['system_cost'].min(),df['system_cost'].max()]
    sns_plot.map_lower(plot_lower)
    sns_plot.map_lower(plot_optimum)
    sns_plot.axes[0,0].set_ylim((0.5,1))
    sns_plot.axes[1,0].set_xlim((0.5,1))
    #sns_plot.map_lower(lambda xdata, ydata, **kwargs:plt.gca().vlines(co2_emis_levels[kwargs['label']],ymin=cost_limits[0],ymax=cost_limits[1],colors=kwargs['color']))
    if save:
        sns_plot.savefig(f'graphics/cost_vs_co2_{prefix}.jpeg')


plot_cost_vs_co2(save=False)

#%%################## corrolelogram tech energy #######################
#######################################################################

def plot_corrolelogram_tech_energy(prefix='',store=False,
    technologies = {'offwind':['offwind','offwind-ac','offwind-dc'],
                'onwind':['onwind'],
                'ror':['ror'],
                'solar':['solar','solar rooftop'],
                'uranium':['uranium'],
                'fossil fuels':['gas','coal','lignite','oil']}):

    df = df_tech_sum[['year']]
    for key in technologies: 
        df[key] = df_tech_e_sum[technologies[key]].sum(axis=1)


    sns_plot = sns.pairplot(df, kind="hist", diag_kind='hist',hue='year',
                            plot_kws=dict(bins=30),
                            diag_kws=dict(bins=40),
                            palette='Set1')

#%%################## Corrolelogram tech ##############################
#######################################################################


def plot_corrolelogram_tech_cap(prefix='',save=False,\
                                technologies={'wind':['wind'],
                                              'lignite + coal' : ['lignite','coal'],
                                              'OCGT + CCGT': ['OCGT','CCGT'],
                                              'solar':['solar','solar rooftop'],
                                              'H2':['H2 Fuel Cell','H2 Electrolysis','H2_store'],
                                              'battery':['battery charger','battery discharger','battery_store']}):

    df = df_tech_sum[['year']]
    for key in technologies: 
        df[key] = df_tech_sum[technologies[key]].sum(axis=1)


    sns_plot = sns.pairplot(df, kind="hist", diag_kind='hist',hue='year',
                            plot_kws=dict(bins=30),
                            diag_kws=dict(bins=40),
                            palette='Set1')

    #plt.suptitle('Scenarios with less than 150% local emisons compared to 100% coal production')
    #plt.suptitle('Scenarios where all countries have more than 10% fosil fuel backup')

    if save:
        sns_plot.savefig(f'graphics/corrolelogram_tech_cap_{prefix}.jpeg')

    fig = sns_plot.fig
    fig.show()


plot_corrolelogram_tech_cap()


#%%######### correlation matrix tech ######################
###########################################################

def plot_correlation_matrix():

    plt.subplots(figsize=(15,15))

    sns.heatmap(df_link_sum.corr(), annot=False,cmap="YlGnBu")
    plt.show()

#%%############ bar plot plot tech ######################
#####################################################

def plot_bar_tech():

    #df_tech_sum = df_tech_sum.replace([np.inf, -np.inf], np.nan)
    #df_tech_sum = df_tech_sum.fillna(0)

    df = df_tech_sum#df_tech_sum[['CCGT','residential rural gas boiler','year']]

    df_long = df.melt(id_vars=['year'],var_name='tech',value_name='capacity')

    #df_long = df_long.replace([np.inf, -np.inf], np.nan)
    #df_long = df_long.fillna(0)

    f, ax = plt.subplots(figsize=(10,30))
    ax.set_xscale("log")
    #sns.despine(bottom=True, left=True)

    # Show each observation with a scatterplot
    sns.boxplot(x="capacity", y="tech", hue="year",
                    fliersize=0,
                data=df_long, )
    plt.xlim(10)

    #plt.savefig(f'graphics/tech_box_{run_name}.jpeg')

#%%############### Corrolelogram secondary metrics ####################
#######################################################################

def plot_corrolelogram_secondary(prefix='',save=False):
    # Autoarky is calculated as the mean self-sufficiency for evvery hour for every country 
    # Gini is calculated using relative energy produvtion vs relative energy consumption 
    # Gini co2 is calculated as relative co2 emission vs 

    df_secondary['transmission'] = df_sum['transmission']
    #df = sns.load_dataset('iris')
    df = df_secondary[['cost_increase','gini_co2','gini','autoarky']]
    df['co2_reduction'] = df['co2_reduction'] = 100 - df_co2.sum(axis=1)/base_emis * 100
    df['year'] = df_secondary['year']
    #df = df[df_secondary.cost_increase<7]
    #df = df[df_secondary.co2_reduction>50]
    #df['co2<150p'] = filt_co2_150p
    #df['10p_backup'] = filt_backup
    #df['c'] = theta.c
    #df['filt_cost'] = filt_cost



    sns_plot = sns.pairplot(df, kind="hist", diag_kind='hist',hue='year',
                                                plot_kws=dict(bins=30),
                                                diag_kws=dict(bins=40),
                                                palette='Set1')
    #plt.suptitle('Scenarios with less than 150% emisons compared to 100% coal production')
    #plt.suptitle('Scenarios where all countries have more than 10% fosil fuel backup')

    if save:
        sns_plot.savefig(f'graphics/secondary_{run_name}.jpeg')
        sns_plot.fig.show()
    
plot_corrolelogram_secondary()

#%% Plot of chain development over time 
def plot_chain_development(prefix='',save=False):
    accept_percent = sum(df_chain.a)/df_theta.shape[0]*100
    print(f'Acceptance {accept_percent:.1f}%')

    df = df_theta
    df['index'] = df.index
    df['s'] = df_chain['s']
    df['c'] = df_chain['c']

    theta_long = pd.wide_to_long(df,stubnames=[''],i='index',j='theta')
    theta_long = theta_long.reset_index()

    #sns.set_theme(style="ticks")
    # Define the palette as a list to specify exact values
    #palette = sns.color_palette("rocket", as_cmap=True)

    #f, ax = plt.subplots(figsize=(10,30))
    # Plot the lines on two facets
    sns.relplot(
        data=theta_long.query('s < 10000 & c <=10 '),
        x="s", y="",
        hue="theta",
        palette=palette,
        row='c',
        ci=None,
        kind="line",
        height=5, aspect=1.5,)
    if save:
        sns_plot.savefig(f'graphics/chain_development_{run_name}.jpeg')
        sns_plot.fig.show()

#%% plot of co2 emis 

def plot_country_co2_emis(prefix='',save=False,\
            countries=['AT','DE','DK','ES','FR','GB','IT','PL']):
#df = df_co2[['DE','DK','FR','PL','ES']]
#df['year'] = df_co2['year']
# 
    df = df_co2[countries+['year']]
    #df = df.sample(100)

    df_long = df.melt(id_vars=['year'],var_name='country',value_name='CO2 emission')

    #sns.set_palette('Paired')
    sns_plot = sns.displot(df_long,x='CO2 emission',
                            kind='kde',
                            #gridsize=100,
                            #bw_adjust=100,
                            common_norm=True,
                            log_scale=True,
                            multiple="stack",
                            hue='country',
                            row='year',
                            palette='Set2',)
    plt.xlim(10**4)

    #df['filt_cost'] = filt_cost


    #sns_plot.map_lower(sns.regplot)
    #sns_plot.savefig('test2.pdf')
    if save:
        sns_plot.savefig(f'graphics/co2_emissions_{prefix}.jpeg')
        sns_plot.fig.show()

plot_country_co2_emis()
#%% plot of thetas 


def plot_thetas(save=False):
    df = df_theta[['1','2','3','4','year']]

    sns_plot = sns.pairplot(df, kind="hist", diag_kind='hist',hue='year',palette='Set2',)
    plt.suptitle('theta values (fraction of CO2 budget)')

    #sns_plot.map_lower(sns.regplot)
    #sns_plot.savefig('test2.pdf')
    if save:
        sns_plot.savefig(f'graphics/thetas_{run_name}.jpeg')
        sns_plot.fig.show()


#%%##################################################
############### test section ########################

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




def nodal_costs(network):

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




    opt_name = {"Store": "e", "Line" : "s", "Transformer" : "s"}
    label = 'test'
    nodal_costs = pd.Series()

    assign_locations(network)
    nodal_costs = calculate_nodal_costs(network,nodal_costs)

    return nodal_costs 

# %%


def calc_150p_coal_emis(network,emis_factor=1.5):
    # Calculate the alowable emissions, if countries are constrained to not emit more co2 than 
    # the emissions it would take to cover 150% of the country demand with coal power 

    # data source https://ourworldindata.org/grapher/carbon-dioxide-emissions-factor
    # 403.2 kg Co2 pr MWh
    co2_emis_pr_ton = 0.45 # ton emission of co2 pr MWh el produced by coal
    country_loads = network.loads_t.p.groupby(network.buses.country,axis=1).sum()
    country_alowable_emis = country_loads.mul(network.snapshot_weightings,axis=0).sum()*co2_emis_pr_ton*emis_factor

    return country_alowable_emis
# %%

network = networks[2030]

bus_total_prod = network.generators_t.p.sum().groupby(network.generators.location).sum()

ac_buses = network.buses.query('carrier == "AC"').index
filt = network.links.bus1.isin(ac_buses) & network.links.carrier.isin(generator_link_carriers)

bus_total_prod += -network.links_t.p1.sum()[filt].groupby(network.links.location).sum()
bus_total_prod.pop('')

load_total= network.loads_t.p_set.sum()
load_total = load_total.groupby(network.buses.country).sum()


rel_demand = load_total/sum(load_total)
rel_generation = bus_total_prod/sum(bus_total_prod)

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

# %%
