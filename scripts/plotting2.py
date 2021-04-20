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
from iso3166 import countries as iso_countries
import plotly.graph_objects as go

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
override_component_attrs["StorageUnit"].loc["p_dispatch"] = ["series","MW",0.,"Storage discharging.","Output"]
override_component_attrs["StorageUnit"].loc["p_store"] = ["series","MW",0.,"Storage charging.","Output"]

#%%#################### import datasets ####################################
############################################################################

prefix = 'elec_open'
sector = 'e'

try :
    pypsa.Network('data/networks/elec_s_37_lv1.5__Co2L0p50-3H-solar+p3-dist1_2030.nc',
                            override_component_attrs=override_component_attrs)
except : 
    os.chdir('..')
    pypsa.Network('data/networks/elec_s_37_lv1.5__Co2L0p50-3H-solar+p3-dist1_2030.nc',
                            override_component_attrs=override_component_attrs)

years = ['2030','2030_o']

load_sweep_data = True

emis_1990 =   1481895952.8
emis_1990_H = 2206933437.8055553

if sector == 'e':
    base_emis = emis_1990
elif sector == 'H':
    base_emis = emis_1990_H


df_names = dict(df_sum='result_sum_vars.csv',
                df_secondary='result_secondary_metrics.csv',
                df_gen_p='result_gen_p.csv',
                df_gen_e='result_gen_E.csv',
                df_co2='result_co2_pr_node.csv',
                df_chain='result_df_chain.csv',
                df_links='result_links_p.csv',
                df_links_E='result_links_E.csv',
                df_lines='result_lines_p.csv',
                df_lines_E='result_lines_E.csv',
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
df_links_E = pd.DataFrame()
df_lines=pd.DataFrame()
df_lines_E = pd.DataFrame()
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
    if sector == 'e':
        run_name = f'mcmc_{year}'
    else :
        run_name = f'mcmc_{year}_H'
    #run_name = 'h99_model'
    networks[year] = pypsa.Network(f'results/{run_name}/network_c0_s1.nc',override_component_attrs=override_component_attrs)

    for df_name in df_names.keys():
        if df_name =='df_nodal_cost':
            df = pd.read_csv(f'results/{run_name}/'+df_names[df_name],index_col=0,header=[0,1,2,3])
        else : 
            df = pd.read_csv(f'results/{run_name}/'+df_names[df_name],index_col=0)
        if df_name == 'df_chain':
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

years = ['2030','2030_o']
if load_sweep_data :
    networks_sweep = {}
    for year in years:
        run_name = f'sweep_{sector}_{year[:4]}'
        #run_name = 'h99_model'
        networks_sweep[year] = pypsa.Network(f'results/{run_name}/network_c0_s1.nc',override_component_attrs=override_component_attrs)

        for df_name in df_names.keys():
            if df_name =='df_nodal_cost':
                df = pd.read_csv(f'results/{run_name}/'+df_names[df_name],index_col=0,header=[0,1,2,3])
            else : 
                try : 
                    df = pd.read_csv(f'results/{run_name}/'+df_names[df_name],index_col=0)
                except: 
                    print(df_name,' not found')

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

def assign_locations(n):
    for c in n.iterate_components(n.one_port_components|n.branch_components):

        ifind = pd.Series(c.df.index.str.find(" ",start=4),c.df.index)

        for i in ifind.unique():
            names = ifind.index[ifind == i]

            if i == -1:
                c.df.loc[names,'location'] = ""
                c.df.loc[names,'country'] = ""
            else:
                c.df.loc[names,'location'] = names.str[:i]
                c.df.loc[names,'country'] = names.str[:2]
    return n 


def set_link_locataions(network):
    network.links['location'] = ""
    network.generators['location'] = ""
    network.lines['location'] = ""
    network.stores['location'] = ""
    #network.storage_units['location']
    query_string = lambda x : f'bus0 == "{x}" | bus1 == "{x}" | bus2 == "{x}" | bus3 == "{x}" | bus4 == "{x}"'
    #id_co2_links = network.links.query(query_string('co2 atmosphere')).index

    country_codes = network.buses.country.unique()
    country_codes = country_codes[:-1]

    # Find all busses assosiated with the model countries 
    country_buses = {code : [] for code in country_codes}
    for country in country_codes:
        country_nodes = list(network.buses.query('country == "{}"'.format(country)).index)
        for b in country_nodes:
            country_buses[country].extend(list(network.buses.query('location == "{}"'.format(b)).index))
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

def remove_duplicates(df):
    index = df.index
    is_duplicate = index.duplicated(keep="first")
    not_duplicate = ~is_duplicate
    df = df[not_duplicate]
    return df

def create_networks_dataframes(networks):
    # df with generators and links from all networks
    
    for key in networks:
        networks[key] = assign_locations(networks[key])

    generators = pd.concat([networks[n].generators for n in networks]).drop_duplicates()
    generators = remove_duplicates(generators)
    links = pd.concat([networks[n].links for n in networks]).drop_duplicates()
    links = remove_duplicates(links)

    stores = pd.concat([networks[n].stores for n in networks]).drop_duplicates()
    stores = remove_duplicates(links)

    return generators, links, stores

def calc_autoarky(generators,links):
    # Soverignity 
    bus_total_prod = df_gen_e.groupby(generators.bus,axis=1).sum().groupby(network.buses.country,axis=1).sum()
    ac_buses = network.buses.query('carrier == "AC"').index
    generator_link_carriers = ['OCGT', 'CCGT', 'coal', 'lignite', 'nuclear', 'oil']
    filt = links.bus1.isin(ac_buses) & links.carrier.isin(generator_link_carriers)
    link_prod = df_links_E[filt.index].loc[:,filt].groupby(links.location,axis=1).sum()
    link_prod[''] = 0

    bus_total_prod += link_prod
    bus_total_prod.pop('')
    bus_total_load = network.loads_t.p.sum().groupby(network.buses.country).sum()
    bus_prod_vs_load = bus_total_prod.divide(bus_total_load)
    bus_prod_vs_load['year'] = df_chain['year']
    #plot_box(bus_prod_vs_load)
    autoarky = bus_prod_vs_load.iloc[:,:33].mean(axis=1)
    return autoarky

def calc_gini(df):
    # Calculates gini coeff

    if df.shape[1] != 33 : 
        print('WARNING dataframe has wrong lenght')

    g_series = pd.Series(index=df.index)
    for row in df.iterrows():
        
        data = row[1]/sum(row[1])
        idy = np.argsort(data)
        data_sort = data[idy]

        # Calculate cumulative sum and add [0,0 as point
        data_sort = np.cumsum(data_sort)
        data_sort = np.concatenate([[0],data_sort])

        step_length = 1/len(data_sort)
        lorenz_integral= 0
        for i in range(len(data_sort)-1):
            lorenz_integral += data_sort[i+1]*step_length

        gini = 1- 2*lorenz_integral
        g_series[row[0]] = gini

    return g_series

def calc_co2_pr_gdp():
    model_countries = network.buses.country.unique()[:33]
    alpha3 = [iso_countries.get(c).alpha3 for c in model_countries]
    df_gdp_i = df_gdp.set_index('Country Code')
    model_countries_gdp = pd.DataFrame(df_gdp_i.loc[alpha3]['2018'])
    model_countries_gdp.index = model_countries

    co2 = df_co2.iloc[:,:33]
    co2_pr_gdp = co2.divide(model_countries_gdp['2018'],axis=1)

    return co2_pr_gdp

def calc_co2_pr_pop():
    model_countries = network.buses.country.unique()[:33]
    alpha3 = [iso_countries.get(c).alpha3 for c in model_countries]
    df_pop_i = df_pop.set_index('Country Code')
    model_countries_pop = pd.DataFrame(df_pop_i.loc[alpha3]['2018'])
    model_countries_pop.index = model_countries

    co2 = df_co2.iloc[:,:33]
    co2_pr_pop = co2.divide(model_countries_pop['2018'],axis=1)

    return co2_pr_pop

def update_secondary_data(df_secondary):
# Calc data for cost increase and co2 reduction 

    cost_increase = (df_secondary.system_cost-network.objective_optimum)/network.objective_optimum*100

    df_secondary['cost_increase'] = cost_increase
    df_secondary['co2_reduction'] = 100 - df_co2.sum(axis=1)/base_emis * 100

    autoarky = calc_autoarky(generators,links)
    df_secondary['autoarky'] = autoarky

    co2_pr_gdp = calc_co2_pr_gdp()
    gini_co2_pr_gdp = calc_gini(co2_pr_gdp)
    df_secondary['gini_co2_pr_gdp'] = gini_co2_pr_gdp

    co2_pr_pop = calc_co2_pr_pop()
    gini_co2_pr_pop = calc_gini(co2_pr_pop)
    df_secondary['gini_co2_pr_pop'] = gini_co2_pr_pop

    return df_secondary

generators, links, stores = create_networks_dataframes(networks)
df_secondary = update_secondary_data(df_secondary)
if df_secondary_sweep.size != 0: 
    df_secondary_sweep = update_secondary_data(df_secondary_sweep)


def create_tech_sum_df(networks,df_links,df_sum,df_store_P,df_gen_e):
    #network = networks[2030]
    # Dataset with with aggregated technology capacities
    #network = set_link_locataions(network)
    
    #idx = links.query('location != "EU" & location != ""').index
    #df_link_sum = df_links[idx].groupby(links.carrier,axis=1).sum()

    links_carrier = pd.concat((networks[key].links.carrier for key in networks))
    links_carrier = links_carrier[~links_carrier.index.duplicated()]
    df_link_sum = df_links.groupby(links_carrier,axis=1).sum()

    stores_carrier = pd.concat((networks[key].stores.carrier for key in networks))
    stores_carrier = stores_carrier[~stores_carrier.index.duplicated()]
    df_store_sum = df_store_P.groupby(stores_carrier,axis=1).sum()
    df_store_sum.columns = [c + '_store' for c in df_store_sum.columns]
    #df_gen_sum = df_gen_p.groupby(network.generators.carrier,axis=1).sum()
    df_gen_sum = df_sum
    #df_gen_sum.pop('oil')

    df_tech_sum = pd.concat([df_link_sum,df_gen_sum,df_store_sum],axis=1)
    #df_tech_sum['wind'] = df_tech_sum[['offwind','offwind-ac','offwind-dc','onwind']].sum(axis=1)

    # Dataset with aggregated technology energy production 
    #df_link_e_sum = df_links_E[idx].groupby(links.carrier,axis=1).sum()

    df_link_e_sum = df_links_E.groupby(links_carrier,axis=1).sum()

    df_store_e_sum = df_store_E.groupby(stores_carrier,axis=1).sum()
    df_store_e_sum.columns = [c + '_store' for c in df_store_e_sum.columns]

    df_gen_e_sum = df_gen_e.groupby(generators.carrier,axis=1).sum()

    df_tech_e_sum = pd.concat([df_link_e_sum,df_gen_e_sum,df_store_e_sum],axis=1)

    return df_tech_sum, df_tech_e_sum

df_tech_sum, df_tech_e_sum = create_tech_sum_df(networks,df_links,df_sum,df_store_P,df_gen_e)
if df_secondary_sweep.size != 0: 
    df_tech_sum_sweep, df_tech_e_sum_sweep = create_tech_sum_df(networks_sweep,df_links_sweep,df_sum_sweep,df_store_P_sweep,df_gen_e_sweep)


#%% Link energy balance in each country

def set_multiindex(df,component):
    index = ((n,component.country[n],component.carrier[n]) for n in df.columns)
    m_index = pd.MultiIndex.from_tuples(index)
    df.columns = m_index

set_multiindex(df_links_E,network.links)
set_multiindex(df_gen_e,network.generators)
set_multiindex(df_storage_E,network.storage_units)
#node_energy = df_links_E.groupby(level=[1,2],axis=1).sum()

# Filter any non electricity producting generators out of the df_gen_e dataframe 
generator_el_energy = df_gen_e.loc[:,(slice(None),df_gen_e.columns.get_level_values(1) != '',slice(None))]

energy_generating_links = ['OCGT','H2 Fuel Cell','battery discharger','home battery discharger','CCGT','coal','lignite','nuclear','oil']
energy_consuming_links = ['H2 Electrolysis','battery charger','Sabatier','helmeth','home battery charger']
energy_distributing_links = ['DC','H2 pipeline','electricity distribution grid'] 

# Multiply generating links with their efficiency 
link_generators_energy = df_links_E.loc[:,(slice(None),slice(None),energy_generating_links)] 
eff = network.links.loc[link_generators_energy.columns.get_level_values(0)].efficiency.values
link_generators_energy = link_generators_energy*eff
link_consumors_energy = - df_links_E.loc[:,(slice(None),slice(None),energy_consuming_links)] 

df_energy = pd.concat((link_consumors_energy,link_generators_energy,df_storage_E,generator_el_energy),axis=1)

df_country_energy = df_energy.groupby(level=[1],axis=1).sum()

df_country_load = network.loads_t.p.groupby(network.buses.country,axis=1).sum().sum()

df_country_k = df_country_energy/df_country_load

df_country_export = df_country_energy-df_country_load

df_energy_dependance =  df_country_export[df_country_export>0].sum(axis=1)

df_country_cost = df_nodal_cost.groupby(level=[2],axis=1).sum().groupby(network.buses.country,axis=1).sum()
df_country_cost = df_country_cost.iloc[:,1:]

#%%##########################################
# ############ plots ########################
# ###########################################
# ####### Corrolelogram cost vs co2 #########

def plot_cost_vs_co2(prefix='',save=False,title= 'Cost vs emission reduction',plot_sweep=False):

    co2_emis_levels = {'2030':1-0.45,
                        '2030_o':1-0.45,
                        2040:1-0.225,
                        2050:1-0.05}

    #run_name = 'elec'
    mga_slack = 0.2

    df = df_secondary[['system_cost']]
    #df['co2 emission'] = df_co2.sum(axis=1)
    df['co2 emission'] =df_store_P['co2 atmosphere']
    df['co2 reduction'] = 1-(df['co2 emission']/base_emis )
    #df['co2 reduction'] = (1-df_co2.sum(axis=1)/base_emis)*100
    df['year'] = df_chain['year']

    cost_limits = [df['system_cost'].min(),df['system_cost'].max()]
    co2_limits = [df['co2 reduction'].min(),df['co2 reduction'].max()]

    def plot_optimum(xdata,ydata,**kwargs):
        plt.gca()
        sns.lineplot(data=df_optimal,
                            x='co2 reduction',
                            y='system_cost',
                            hue='year',
                            palette='Set2' #'Set1'
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

    sns_plot = sns.pairplot(df, 
                            vars=['co2 reduction','system_cost'],
                            kind="hist", 
                            diag_kind='hist',
                            hue='year',
                            plot_kws=dict(bins=60),
                            diag_kws=dict(bins=60,kde=False,log_scale=False),
                            aspect=1.6,
                            height=3,
                            palette='Set2')
    plt.suptitle(title)
    #plt.suptitle('Scenarios where all countries have more than 10% fosil fuel backup')

    cost_limits = [df['system_cost'].min(),df['system_cost'].max()]
    sns_plot.map_lower(plot_lower)
    if plot_sweep:
        df_optimal = df_secondary_sweep[['system_cost']]
        df_optimal['co2 emission'] = df_store_P_sweep['co2 atmosphere']
        df_optimal['co2 reduction'] = 1-(df_optimal['co2 emission']/base_emis )
        df_optimal['year'] = df_co2_sweep['year']
        sns_plot.map_lower(plot_optimum)
    sns_plot.axes[0,0].set_ylim((0.45,1))
    sns_plot.axes[1,1].set_ylim(0.5e11,0.7e11)
    sns_plot.axes[1,0].set_xlim((0.45,0.85))
    #sns_plot.map_lower(lambda xdata, ydata, **kwargs:plt.gca().vlines(co2_emis_levels[kwargs['label']],ymin=cost_limits[0],ymax=cost_limits[1],colors=kwargs['color']))
    if save:
        sns_plot.savefig(f'graphics/cost_vs_co2_{prefix}.jpeg')


plot_cost_vs_co2(save=True,prefix=prefix,plot_sweep=True)

#%%################## corrolelogram tech energy #######################
#######################################################################

def plot_corrolelogram_tech_energy(prefix='',save=False,
    technologies = {'wind':['offwind','offwind-ac','offwind-dc','onwind'],
                    'lignite + coal' : ['lignite','coal'],
                    'OCGT + CCGT': ['OCGT','CCGT'],
                    'solar':['solar','solar rooftop'],
                    'H2':['H2 Fuel Cell','H2 Electrolysis','H2_store'],
                    'battery':['battery charger','battery discharger','battery_store']},
                    title='Technology energy production'):

    df = df_chain[['year']]
    for key in technologies: 
        df[key] = df_tech_e_sum[technologies[key]].sum(axis=1)


    sns_plot = sns.pairplot(df, kind="hist", diag_kind='hist',hue='year',
                            plot_kws=dict(bins=30),
                            diag_kws=dict(bins=40,),
                            palette='Set2')

    plt.suptitle(title)
    #plt.legend(labels=['test'])

    if save:
        sns_plot.savefig(f'graphics/corrolelogram_tech_energy_{prefix}.jpeg')

plot_corrolelogram_tech_energy(prefix=prefix,save=True)


#%%################## Corrolelogram tech ##############################
#######################################################################


def plot_corrolelogram_tech_cap(prefix='',save=False,\
                                technologies={'wind':['offwind','offwind-ac','offwind-dc','onwind'],
                                              'lignite + coal' : ['lignite','coal'],
                                              'OCGT + CCGT': ['OCGT','CCGT'],
                                              'solar':['solar','solar rooftop'],
                                              'H2':['H2 Fuel Cell','H2 Electrolysis','H2_store'],
                                              'battery':['battery charger','battery discharger','battery_store']},\
                                title = 'Technology capacities'):

    df = df_chain[['year']]
    for key in technologies:
        df[key] = df_tech_sum[technologies[key]].sum(axis=1)

    sns_plot = sns.pairplot(df, kind="hist", diag_kind='hist',hue='year',
                            plot_kws=dict(bins=30),
                            diag_kws=dict(bins=40),
                            palette='Set2')


    df_sweep = df_chain[['year']]
    for key in technologies: 
        df_sweep[key] = df_tech_sum_sweep[technologies[key]].sum(axis=1)

    # Remove last point as this is an extreme 
    df_sweep = df_sweep.iloc[:-1,:]
    def plot_lower(xdata,ydata,**kwargs):
        year = 2050
        ax = plt.gca()
        sns.scatterplot(x = df_sweep.query(f'year == {year}')[xdata.name],
                    y = df_sweep.query(f'year == {year}')[ydata.name],
                    hue = -df_co2_sweep.query(f'year == {year}').iloc[:,:33].sum(axis=1),
                    markers=['X'],
                    size=30,
                    palette='rocket',
                    ax=ax)


    sns_plot.map_offdiag(plot_lower)

    plt.suptitle(title)
    #plt.legend(labels=['test'])

    if save:
        sns_plot.savefig(f'graphics/corrolelogram_tech_cap_{prefix}.jpeg')

    fig = sns_plot.fig
    fig.show()


plot_corrolelogram_tech_cap(prefix=prefix,save=True)


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

plot_bar_tech()

#%%############### Corrolelogram secondary metrics ####################
#######################################################################

def plot_corrolelogram_secondary(prefix='',save=False,\
                             metrics={'cost increase':['cost_increase'],
                                      'co2 reduction':['co2_reduction'],
                                      #'gini production vs consumption':['gini'],
                                      'gini co2 vs pop':['gini_co2_pr_pop'],
                                      #'gini co2':['gini_co2'],
                                      'autoarky':['autoarky']},
                                title = 'Secondary metrics',
                                plot_optimum = False):
    # Autoarky is calculated as the mean self-sufficiency for evvery hour for every country 
    # Gini is calculated using relative energy production vs relative energy consumption 
    # Gini co2 is calculated as relative co2 emission vs 

    df = df_chain[['year']]
    for key in metrics: 
        df[key] = df_secondary[metrics[key]].sum(axis=1)

    sns_plot = sns.pairplot(df, kind="hist", diag_kind='hist',hue='year',
                                                plot_kws=dict(bins=30),
                                                diag_kws=dict(bins=40,alpha=0.5),
                                                #palette='RdYlBu'
                                                palette='Set2'
                                                )


    if plot_optimum:
        #df_sweep = df_secondary_sweep[['year']]
        #for key in metrics: 
        #    df_sweep[key] = df_secondary_sweep[metrics[key]].sum(axis=1)
        
        optimum_index = np.where(df_chain.s == 1)[0][0]

        df_secondary.iloc[[17],:]
        df_optimum = df_chain.loc[[optimum_index],['year']]
        for key in metrics:
            df_optimum[key] = df_secondary.loc[optimum_index,metrics[key]].sum()

        def plot_lower(xdata,ydata,**kwargs):
            year = 2050
            ax = plt.gca()
            plt.scatter(x = df_optimum[xdata.name],
                        y = df_optimum[ydata.name],
                        c='red',
                        s=200,
                        #hue = -df_co2_sweep.query(f'year == {year}').iloc[:,:33].sum(axis=1),
                        marker='X',
                        #sizes=[50],
                        #palette='rocket',
                        #ax=ax
                        )

        sns_plot.map_offdiag(plot_lower)

    plt.suptitle(title)

    #sns_plot.axes[0,0].set_ylim(0,1)

    if save:
        sns_plot.savefig(f'graphics/secondary_{prefix}.jpeg')
        sns_plot.fig.show()
    
df_secondary['energy dependance'] = df_energy_dependance 
plot_corrolelogram_secondary(save=True,prefix=prefix,plot_optimum=True,
                            metrics={'cost increase':['cost_increase'],
                                      'co2 reduction':['co2_reduction'],
                                      #'gini production vs consumption':['gini'],
                                      'gini co2 vs pop':['gini_co2_pr_pop'],
                                      #'gini co2':['gini_co2'],
                                      #'autoarky':['autoarky'],
                                      'energy dependance':['energy dependance']})

#%%#########################################################################
################################# geo plot ########################################

def plot_geo(df,title):
    # input should be pandas seres or dataframe with one column af data values 
    # index must be alpha2 country codes 

    alpha3_index = [iso_countries.get(c).alpha3 for c in df.index]

    fig = go.Figure()

    fig.add_trace(go.Choropleth(
                        geo='geo1',
                        locations = alpha3_index,
                        z = list(df.values),#/area,
                        text = alpha3_index,
                        colorscale = 'Thermal',
                        #autocolorscale=False,
                        #zmax=283444,
                        #zmin=0,
                        #reversescale=False,
                        #marker_line_color='darkgray',
                        #marker_line_width=0.5,
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
        title=title,
        width=900,
        height=500,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )


    fig.show()
    return fig

#%%################### box plot ############################
###############################################################

def plot_box(df_wide,df_wide_optimal=None,prefix='',save=False,title='',name='co2_box'):
    #df_wide = co2_pr_pop
    model_countries = network.buses.country.unique()[:33]
    df = pd.melt(df_wide,value_vars=model_countries,id_vars='year',var_name='Country')
    #df = df.query('year == 2030 | year == 2050')

    f,ax = plt.subplots(figsize=(30,10))
    sns_plot = sns.boxplot(x='Country', y="value", hue="year",
                        data=df, palette="muted",
                        ax=ax)
    if df_wide_optimal is not None:
        df_optimal = pd.melt(df_wide_optimal,value_vars=model_countries,id_vars='year',var_name='Country')
        sns.stripplot(x='Country',y='value',hue='year',
                        data=df_optimal,
                        jitter=0,
                        color='red',
                        marker='X',
                        size=10,
                        ax=ax)

    plt.ylabel('CO2 emission')
    plt.suptitle(title)

    if save:
        plt.savefig(f'graphics/{name}_{prefix}.jpeg')


#%%

def plot_co2_box():

    #df = df_co2/df_country_energy
    df = df_co2
    df['year'] = df_chain[['year']]

    optimal_index = df_chain.query('s == 1 & c == 0').index
    df_optimal = df.iloc[optimal_index]

    plot_box(df,df_optimal,title='Country co2 emmission',save=True,prefix=prefix,name='co2_box')

plot_co2_box()
#%%

def plot_unused_co2():

    df = df_theta.copy()
    df.columns=mcmc_variables


    for year in networks:
        co2_budget = networks[year].global_constraints.loc['CO2Limit'].constant
        df.loc[df_chain.year == year] = df.loc[df_chain.year == year].multiply(co2_budget)

    df = df.iloc[:,:33] - df_co2.iloc[:,:33]

    df['year'] = df_chain['year']
    plot_box(df,title='Unused Co2',prefix=prefix,name='unused_co2',save=True)

plot_unused_co2()
#%%##################### co2 emis pr gdp geo plot #############
###############################################################

def plot_co2_gpd_geo(prefix='',save=False,year=2030):
    model_countries = network.buses.country.unique()[:33]
    alpha3 = [iso_countries.get(c).alpha3 for c in model_countries]
    df_gdp_i = df_gdp.set_index('Country Code')
    model_countries_gdp = pd.DataFrame(df_gdp_i.loc[alpha3]['2018'])
    model_countries_gdp.index = model_countries

    co2_pr_gdp = df_co2.divide(model_countries_gdp['2018'],axis=1)
    co2_pr_gdp['year'] = df_chain['year']


    df = co2_pr_gdp.query(f'year == {year}').mean()[:33]*1e6

    fig = plot_geo(df)
    if save:
        fig.write_image(f'graphics/co2_gdp_geo_{prefix}.jpeg')


#%%################# co2 emis pr pop geo plot ##############################
############################################################################


def plot_co2_pop_geo(prefix='',save=False,year=2030,title='Mean co2 emis pr population'):
    model_countries = network.buses.country.unique()[:33]
    alpha3 = [iso_countries.get(c).alpha3 for c in model_countries]
    df_pop_i = df_pop.set_index('Country Code')
    model_countries_pop = pd.DataFrame(df_pop_i.loc[alpha3]['2018'])
    model_countries_pop.index = model_countries

    co2_pr_pop = df_co2.divide(model_countries_pop['2018'],axis=1)
    co2_pr_pop['year'] = df_chain['year']


    df = co2_pr_pop.query(f'year == {year}').mean()[:33]*1e6

    fig = plot_geo(df,title)
    if save:
        fig.write_image(f'graphics/co2_pop_geo_{year}_{prefix}.jpeg')

plot_co2_pop_geo(year=2050,save=True,prefix=prefix,title='Mean co2 emis pr population 2050')


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
        palette='Set2',
        row='c',
        ci=None,
        kind="line",
        height=5, aspect=1.5,)
    if save:
        sns_plot.savefig(f'graphics/chain_development_{run_name}.jpeg')
        sns_plot.fig.show()

plot_chain_development()

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
#mcmc_variables = read_csv(f'results/{run_name}/mcmc_variables.csv')

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


def calculate_energy(n,energy):

    for c in n.iterate_components(n.one_port_components|n.branch_components):

        if c.name in n.one_port_components:
            c_energies = c.pnl.p.multiply(n.snapshot_weightings,axis=0).sum().multiply(c.df.sign).groupby(c.df.carrier).sum()
        else:
            c_energies = pd.Series(0.,c.df.carrier.unique())
            for port in [col[3:] for col in c.df.columns if col[:3] == "bus"]:
                totals = c.pnl["p"+port].multiply(n.snapshot_weightings,axis=0).sum()
                #remove values where bus is missing (bug in nomopyomo)
                no_bus = c.df.index[c.df["bus"+port] == ""]
                try :
                    totals.loc[no_bus] = n.component_attrs[c.name].loc["p"+port,"default"]
                except : 
                    pass
                c_energies -= totals.groupby(c.df.carrier).sum()

        c_energies = pd.concat([c_energies], keys=[c.list_name])

        energy = energy.reindex(c_energies.index|energy.index)

        energy.loc[c_energies.index] = c_energies

    return energy

def assign_locations(n):
    for c in n.iterate_components(n.one_port_components|n.branch_components):

        ifind = pd.Series(c.df.index.str.find(" ",start=4),c.df.index)

        for i in ifind.unique():
            names = ifind.index[ifind == i]

            if i == -1:
                c.df.loc[names,'location'] = ""
                c.df.loc[names,'country'] = ""
            else:
                c.df.loc[names,'location'] = names.str[:i]
                c.df.loc[names,'country'] = names.str[:2]


opt_name = {"Store": "e", "Line" : "s", "Transformer" : "s"}
label = 'test'
energy = pd.Series()

assign_locations(network)
energy = calculate_energy(network,energy)


# %%
