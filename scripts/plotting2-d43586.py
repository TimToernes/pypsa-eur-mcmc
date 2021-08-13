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

prefix = '2030_elec'
sector = 'e'
burnin_samples = 1000

emis_1990 =   1481895952.8
emis_1990_H = 2206933437.8055553

#network_path = 'data/networks/elec_s_37_lv1.5__Co2L0p50-3H-solar+p3-dist1_2030.nc'
network_path = 'results/mcmc_2030/network_c0_s1.nc'

try :
    network = pypsa.Network(network_path,
                override_component_attrs=override_component_attrs)
except : 
    os.chdir('..')
    network = pypsa.Network(network_path,
                override_component_attrs=override_component_attrs)



scenarios = ['mcmc_2030','sweep_2030','scenarios']

load_sweep_data = False

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
                df_theta='result_theta.csv',
                df_nodal_el_price='result_bus_nodal_price.csv',
                df_nodal_co2_price='result_national_co2_dual.csv',
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
for run_name in scenarios:
    #if sector == 'e':
    #    run_name = f'{year}'
    #else :
    #    run_name = f'mcmc_{year}_H'
    #run_name = 'h99_model'
    networks[run_name] = pypsa.Network(f'results/{run_name}/network_c0_s1.nc',override_component_attrs=override_component_attrs)

    for df_name in df_names.keys():
        if df_name =='df_nodal_cost':
            df = pd.read_csv(f'results/{run_name}/'+df_names[df_name],index_col=0,header=[0,1,2,3])
        else : 
            df = pd.read_csv(f'results/{run_name}/'+df_names[df_name],index_col=0)
        if df_name == 'df_chain':
            df['year'] = run_name
        try :
            dfs[df_name] = pd.concat((dfs[df_name],df),ignore_index=True)
            vars()[df_name] = dfs[df_name]
        except Exception:
            dfs[df_name] = df
            vars()[df_name] = dfs[df_name]

network = networks[scenarios[0]]
mcmc_variables = read_csv(networks[run_name].mcmc_variables)
mcmc_variables = [row[0]+row[1] for row in mcmc_variables]
                
df_pop = pd.read_csv('data/API_SP.POP.TOTL_DS2_en_csv_v2_2106202.csv',sep=',',index_col=0,skiprows=3)
df_gdp = pd.read_csv('data/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_2055594.csv',sep=',',index_col=0,skiprows=3)

years = ['2030',]
if load_sweep_data :
    networks_sweep = {}
    for year in years:
        #run_name = f'sweep_{sector}_{year[:4]}'
        run_name = 'sweep_2030'
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


#%%##### Data postprocessing #################
############################################
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
    stores = remove_duplicates(stores)

    storage_units = pd.concat([networks[n].storage_units for n in networks]).drop_duplicates()
    storage_units = remove_duplicates(storage_units)

    return generators, links, stores, storage_units

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

    df_norm = df.div(df.sum(axis=1),axis=0)

    data_sort = df_norm.values.copy()
    data_sort = np.append(data_sort,np.zeros((data_sort.shape[0],1)),axis=1)
    data_sort.sort()

    data_cum = data_sort.cumsum(axis=1)

    lorentz_integrals = np.trapz(data_cum,dx=1/df.shape[1])

    ginies = 1-2*lorentz_integrals

    return pd.Series(ginies)

def calc_gini_old(df):
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

        step_length = 1/(len(data_sort)-1)
        lorenz_integral= 0
        for i in range(len(data_sort)):
            lorenz_integral += data_sort[i]*step_length

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
    df_secondary['co2_emission'] = df_co2.sum(axis=1)
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

generators, links, stores, storage_units = create_networks_dataframes(networks)
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

def create_country_pop_df(df_pop,df_gdp):
    model_countries = network.buses.country.unique()[:33]
    alpha3 = [iso_countries.get(c).alpha3 for c in model_countries]
    df_pop_i = df_pop.set_index('Country Code')
    df_gdp_i = df_gdp.set_index('Country Code')

    model_countries_pop = pd.DataFrame(df_pop_i.loc[alpha3]['2018'])
    model_countries_gdp = pd.DataFrame(df_gdp_i.loc[alpha3]['2018'])
    model_countries_pop.index = model_countries
    model_countries_gdp.index = model_countries
    return pd.Series(model_countries_pop['2018']), pd.Series(model_countries_gdp['2018'])

df_country_pop,df_country_gdp = create_country_pop_df(df_pop,df_gdp)

def calc_country_dfs():

    def set_multiindex(df,component):
        index = ((n,component.country[n],component.carrier[n]) for n in df.columns)
        m_index = pd.MultiIndex.from_tuples(index)
        df.columns = m_index

    set_multiindex(df_links_E,links)
    set_multiindex(df_gen_e,generators)
    set_multiindex(df_storage_E,storage_units)
    #node_energy = df_links_E.groupby(level=[1,2],axis=1).sum()

    # Filter any non electricity producting generators out of the df_gen_e dataframe 
    generator_el_energy = df_gen_e.loc[:,(slice(None),df_gen_e.columns.get_level_values(1) != '',slice(None))]

    energy_generating_links = ['OCGT','H2 Fuel Cell','battery discharger','home battery discharger','CCGT','coal','lignite','nuclear','oil']
    energy_consuming_links = ['H2 Electrolysis','battery charger','Sabatier','helmeth','home battery charger']
    energy_distributing_links = ['DC','H2 pipeline','electricity distribution grid'] 

    # Multiply generating links with their efficiency 
    link_generators_energy = df_links_E.loc[:,(slice(None),slice(None),energy_generating_links)] 
    eff = links.loc[link_generators_energy.columns.get_level_values(0)].efficiency.values
    link_generators_energy = link_generators_energy*eff
    link_consumors_energy = - df_links_E.loc[:,(slice(None),slice(None),energy_consuming_links)] 

    df_energy = pd.concat((link_consumors_energy,link_generators_energy,df_storage_E,generator_el_energy),axis=1)

    df_country_energy = df_energy.groupby(level=[1],axis=1).sum()

    df_country_load = network.loads_t.p.sum().groupby(network.buses.country).sum()

    df_country_k = df_country_energy/df_country_load

    df_country_export = df_country_energy-df_country_load

    df_energy_dependance =  df_country_export[df_country_export>0].sum(axis=1)

    df_country_cost = df_nodal_cost.groupby(level=[2],axis=1).sum().groupby(network.buses.country,axis=1).sum()
    df_country_cost = df_country_cost.iloc[:,1:]

    df_nodal_cost_marginal = df_nodal_cost.loc[:,(slice(None),'marginal',slice(None))]
    df_country_marginal_cost = df_nodal_cost_marginal.groupby(level=[2],axis=1).sum().groupby(network.buses.country,axis=1).sum()
    df_country_marginal_cost = df_country_marginal_cost.iloc[:,1:]

    return df_energy, df_country_energy, df_country_load, df_country_k, df_country_export, df_energy_dependance, df_country_cost, df_country_marginal_cost

df_energy, df_country_energy, df_country_load, df_country_k, df_country_export, df_energy_dependance, df_country_cost, df_country_marginal_cost = calc_country_dfs()

df_secondary['gini_cost_pop'] = calc_gini(df_country_cost/df_country_pop)
df_secondary['gini_cost_energy'] = calc_gini(df_country_cost/df_country_pop)
df_secondary['gini_co2_energy'] = calc_gini(df_country_cost/df_country_pop)
df_secondary['gini_cost'] = calc_gini(df_country_cost)
df_secondary['gini_marginal_cost'] = calc_gini(df_country_marginal_cost)
df_secondary['energy_dependance'] = df_energy_dependance
df_secondary['gini_co2_price'] = calc_gini(df_nodal_co2_price/df_country_pop)

df_country_el_price = df_nodal_el_price[network.buses.query('carrier == "AC"').index].groupby(network.buses.country,axis=1).sum()
df_secondary['gini_el_price'] = calc_gini(df_country_el_price/df_country_pop)

df_theta.columns = mcmc_variables
df_co2_assigned = df_theta*base_emis*0.45

# Filters 
filt_co2_cap = df_co2.sum(axis=1)<=base_emis*0.448
filt_burnin = df_chain.s>burnin_samples


#%%############# plots #############################
# ############################################
# ###########################################
# ####### Corrolelogram cost vs co2 #########

def plot_cost_vs_co2(prefix='',save=False,
                    title= 'Cost vs emission reduction',
                    plot_sweep=False,
                    plot_optimum=False,
                    plot_scenarios=False,
                    sweep_name='sweep_2030'):

    co2_emis_levels = {'2030':1-0.45,
                        '2030_o':1-0.45,
                        '2030_n':1-0.45,
                        '2030f':1-0.45,
                        'mcmc':1-0.45,
                        'swee':1-0.45,
                        2040:1-0.225,
                        2050:1-0.05}

    # Create dataframe with relevant data
    df = df_secondary[['cost_increase']]
    df.rename(columns={'cost_increase':'Cost increase'},inplace=True)
    #df['co2 emission'] = df_co2.sum(axis=1)
    df['co2 emission'] =df_store_P['co2 atmosphere']
    df['CO2 reduction'] = (1-(df['co2 emission']/base_emis ))*100
    #df['co2 reduction'] = (1-df_co2.sum(axis=1)/base_emis)*100
    df['year'] = df_chain['year']
    #df = df[df_chain.a==1]

    # Create dataframe with optimal solution
    if plot_optimum:
        index_optimum = df_chain.query('c==1 & s==1').index
        df_optimum = df.iloc[index_optimum]

    if plot_sweep:
        index_sweep = df_chain.query(f'year == "{sweep_name}"').index
        df_sweep = df.iloc[index_sweep]

    if plot_scenarios:
        index_scenarios = df_chain.query(f'year == "scenarios"').index
        df_scenarios = df.iloc[index_scenarios]
        df_scenarios = df_scenarios.iloc[0:2]


    # filter out burnin samples
    df = df[ filt_burnin & filt_co2_cap]

    cost_limits = [df['Cost increase'].min(),df['Cost increase'].max()]
    co2_limits = [df['CO2 reduction'].min(),df['CO2 reduction'].max()]

    def scenarios_plot(xdata,ydata,**kwargs):
        plt.gca()
        plt.scatter(df_scenarios['CO2 reduction'],
                    df_scenarios['Cost increase'],
                    s=150,
                    c=['g','orange'],
                    marker='^',
                    )

    def sweep_plot(xdata,ydata,**kwargs):
        plt.gca()
        styles = ['bD','ms']
        #for i in range(2):
        plt.plot(df_sweep['CO2 reduction'],
                df_sweep['Cost increase'],
                        #styles[i],
                        #marker='D',
                        #mfc='g',
                        markersize=10)

    # Function for plotting vertical lines for CO2 limit 
    def plot_lower(xdata,ydata,**kwargs):
        plt.gca().vlines(co2_emis_levels['2030']*100,
                        ymin=cost_limits[0],
                        ymax=cost_limits[1],
                        colors = 'r',
                        #colors=kwargs['color']
                        )

    def optimum_plot(xdata,ydata,**kwargs):
        plt.gca()
        plt.plot(df_optimum['CO2 reduction'],
                df_optimum['Cost increase'],
                        marker='X',
                        mfc='r',
                        markersize=20)

    sns_plot = sns.pairplot(df, 
                            vars=['CO2 reduction','Cost increase'],
                            kind="hist", 
                            diag_kind='hist',
                            #hue='year',
                            plot_kws=dict(bins=50,thresh=0),
                            diag_kws=dict(bins=40,kde=False,log_scale=False),
                            aspect=1.6,
                            height=3,
                            palette='Set2')
    plt.suptitle(title)
    #plt.suptitle('Scenarios where all countries have more than 10% fosil fuel backup')

    cost_limits = [df['Cost increase'].min(),df['Cost increase'].max()]
    sns_plot.map_lower(plot_lower)
    if plot_sweep:
        sns_plot.map_lower(sweep_plot)

    # Draw optimal solution on plot 
    if plot_optimum:
        sns_plot.map_lower(optimum_plot)

    if plot_scenarios:
        sns_plot.map_lower(scenarios_plot)

    #sns_plot.axes[0,0].set_ylim((0.45,1))
    #sns_plot.axes[1,1].set_ylim(0.53e11,0.55e11)
    # sns_plot.axes[1,0].set_xlim((0.908,0.92))
#
    #sns_plot.map_lower(lambda xdata, ydata, **kwargs:plt.gca().vlines(co2_emis_levels[kwargs['label'][:4]],ymin=cost_limits[0],ymax=cost_limits[1],colors=kwargs['color']))
    if save:
        sns_plot.savefig(f'graphics/cost_vs_co2_{prefix}.jpeg',dpi=400)


plot_cost_vs_co2(save=True,prefix=prefix,plot_sweep=True,plot_optimum=True,plot_scenarios=True)

#%%################## corrolelogram tech energy #######################
#######################################################################

def plot_corrolelogram_tech_energy(prefix='',save=False,plot_optimum=False,
    technologies = {'wind':['offwind','offwind-ac','offwind-dc','onwind'],
                    'lignite + coal' : ['lignite','coal'],
                    'gas': ['OCGT','CCGT'],
                    'solar':['solar','solar rooftop'],
                    #'H2':['H2 Fuel Cell','H2 Electrolysis','H2_store'],
                    'H2':['H2 Fuel Cell',],
                    #'battery':['battery charger','battery discharger','battery_store']
                    'battery':['battery discharger']
                    },
                    title='Technology energy production'):

    df = df_chain[['year']]
    for key in technologies: 
        #df[key] = df_tech_e_sum[technologies[key]].sum(axis=1)
        df[key] = df_energy.loc[:,(slice(None),slice(None),technologies[key])].sum(axis=1)

    if plot_optimum:
        optimum_index = np.where(df_chain.s == 1)[0][0]
        df_optimum = df.iloc[[17]]


    # filter out burnin samples
    df = df[df_chain.s>burnin_samples]

    sns_plot = sns.pairplot(df, kind="hist", diag_kind='hist',hue='year',
                            plot_kws=dict(bins=30),
                            diag_kws=dict(bins=40,),
                            palette='Set2')

    def plot_lower(xdata,ydata,**kwargs):
        #year = 2050
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
    if plot_optimum:
        sns_plot.map_offdiag(plot_lower)

    plt.suptitle(title)
    #plt.legend(labels=['test'])

    if save:
        sns_plot.savefig(f'graphics/corrolelogram_tech_energy_{prefix}.jpeg')

plot_corrolelogram_tech_energy(prefix=prefix,save=True,plot_optimum=True)


#%%################## Corrolelogram tech ##############################
#######################################################################


def plot_corrolelogram_tech_cap(prefix='',save=False,\
                                technologies={'wind':['offwind','offwind-ac','offwind-dc','onwind'],
                                              'lignite + coal' : ['lignite','coal'],
                                              'OCGT + CCGT': ['OCGT','CCGT'],
                                              'solar':['solar'],
                                              'H2':['H2 Fuel Cell','H2 Electrolysis','H2_store'],
                                              'battery':['battery charger','battery discharger','battery_store']},\
                                title = 'Technology capacities',
                                plot_optimum=False):

    df = df_chain[['year']]
    for key in technologies:
        df[key] = df_tech_sum[technologies[key]].sum(axis=1)


    if plot_optimum:
        optimum_index = np.where(df_chain.s == 1)[0][0]
        df_optimum = df.iloc[[17]]

    # filter out burnin samples
    df = df[filt_burnin & filt_co2_cap]

    sns_plot = sns.pairplot(df, kind="hist", diag_kind='hist',hue='year',
                            plot_kws=dict(bins=30),
                            diag_kws=dict(bins=40),
                            palette='Set2')
    if plot_optimum:#
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
    #plt.legend(labels=['test'])

    if save:
        sns_plot.savefig(f'graphics/corrolelogram_tech_cap_{prefix}.jpeg')

    fig = sns_plot.fig
    fig.show()


plot_corrolelogram_tech_cap(prefix=prefix,save=True,plot_optimum=True)


#%%############### Corrolelogram secondary metrics ####################
#######################################################################

def plot_corrolelogram_secondary(prefix='',save=False,\
                             metrics={'cost increase':['cost_increase'],
                                      'co2 reduction':['co2_reduction'],
                                      #'gini production vs consumption':['gini'],
                                      'gini co2 vs pop':['gini_co2_pr_pop'],
                                      'gini co2 price':['gini_co2_price'],
                                      'autoarky':['autoarky']},
                                title = 'Secondary metrics',
                                plot_optimum = False):
    # Autoarky is calculated as the mean self-sufficiency for evvery hour for every country 
    # Gini is calculated using relative energy production vs relative energy consumption 
    # Gini co2 is calculated as relative co2 emission vs 

    df = df_chain[['year']]
    for key in metrics: 
        df[key] = df_secondary[metrics[key]].sum(axis=1)


    if plot_optimum:
        #optimum_index = np.where(df_chain.s == 1)[0][0]
        optimal_index = df_chain.query('year == "scenarios"').index
        optimal_index = optimal_index.append(df_chain.query('s == 1 & c == 1').index)
        #df_optimal = df.iloc[optimal_index]
        #df_secondary.iloc[[17],:]
        df_optimum = df_chain.iloc[optimal_index][['year']]
        for key in metrics:
            df_optimum[key] = df_secondary.iloc[optimal_index][metrics[key]].sum(axis=1)


    filt_low_co2_red = df_secondary.co2_reduction<=70
    # filter out burnin samples
    df = df[filt_burnin & filt_co2_cap ]
    #df['low_co2'] = ~filt_low_co2_red

    sns_plot = sns.pairplot(df.sample(frac = 0.2), kind="hist", diag_kind='hist',
                                                #hue='low_co2',
                                                plot_kws=dict(bins=30),
                                                #plot_kws=dict(marker="o", linewidth=1,alpha=0.1),
                                                diag_kws=dict(bins=40,alpha=0.5),
                                                #palette='RdYlBu'
                                                palette='Set1'
                                                )

    if plot_optimum:
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
                                      'gini co2 price':['gini_co2_price'],
                                      'gini el price':['gini_el_price'],
                                      'energy dependance':['energy dependance']})


#%%################### box plot ############################
###############################################################

def plot_box(df_wide,df_wide_optimal=None,prefix='',save=False,title='',name='co2_box',ylabel='CO2 emission',ylim=None,**kwargs):
    #df_wide = co2_pr_pop
    model_countries = network.buses.country.unique()[:33]
    df = pd.melt(df_wide,value_vars=model_countries,id_vars='year',var_name='Country')
    #df = df.query('year == 2030 | year == 2050')

    f,ax = plt.subplots(figsize=(11,4.5))
    sns_plot = sns.boxplot(x='Country', y="value", #hue_order=model_countries, #hue="year",
                        data=df, 
                        #palette="muted",
                        order=df_wide.columns[:-1],
                        ax=ax,
                        **kwargs)

    if df_wide_optimal is not None:
        df_optimal = pd.melt(df_wide_optimal,value_vars=model_countries,id_vars=['year','scenario'],var_name='Country')
        sns.stripplot(x='Country',y='value',#hue='Country',
                        data=df_optimal,
                        order=df_wide.columns[:-1],
                        jitter=0,
                        hue='scenario',
                        palette={'local load':'g','local 1990':'orange','Optimum unconstrained':'c','Optimum':'r'},
                        #color='red',
                        #marker='X',
                        size=5,
                        ax=ax,)


    plt.ylabel(ylabel)
    plt.suptitle(title)

    if ylim != None:
        plt.ylim(ylim)

    if save:
        plt.savefig(f'graphics/{name}_{prefix}.jpeg',transparent=False,dpi=400)


#%% Set index order for box plots

#index_order = df_co2.mean().sort_values().index

#index_order = (df_co2/df_country_load).mean().sort_values().index

df = df_co2/df_country_energy
df = df[filt_burnin & filt_co2_cap]

index_order = (df_co2/df_country_energy).mean().sort_values().index

#%%
def plot_co2_box():

    #df = df_co2/df_country_energy
    df = df_co2
    df['year'] = df_chain[['year']]

    # assign new sort order 
    #df = df[df.mean().sort_values().index]
    df = df[index_order]
    df['year'] = df_chain['year']

    optimal_index = df_chain.query('year == "scenarios"').index
    optimal_index = optimal_index.append(df_chain.query('s == 1 & c == 1').index)
    df_optimal = df.iloc[optimal_index]
    #df_optimal = df_optimal.append(df_co2_sweep)
    df_optimal['scenario'] = np.array(['local load','local 1990','Optimum unconstrained','Optimum',])

    # filter out burnin samples
    df = df[filt_burnin & filt_co2_cap]

    plot_box(df,df_optimal,title='Country co2 emmission',save=True,prefix=prefix,name='co2_box')

plot_co2_box()


#%%

def plot_co2_pr_mwh_box():

    df = df_co2/df_country_energy
    df['year'] = df_chain['year']

    # Rearange collumns to match index order 
    df = df[index_order]
    df['year'] = df_chain['year']

    optimal_index = df_chain.query('year == "scenarios"').index
    optimal_index = optimal_index[:2]
    optimal_index = optimal_index.append(df_chain.query('s == 1 & c == 1').index)
    df_optimal = df.iloc[optimal_index]
    df_optimal['scenario'] = np.array(['local load','local 1990','Optimum',])
    # filter out burnin samples
    df = df[filt_burnin & filt_co2_cap]

    plot_box(df,df_optimal,ylabel='T Co2 pr MWh produced',
                           title='Co2 intensity pr MWh',
                           save=True,
                           prefix=prefix,
                           name='co2_mwh_box',
                           fliersize=0.5,
                           linewidth=1,
                           color='tab:blue')
    plt.gca()
#plt.ylim((0,5e4))
plot_co2_pr_mwh_box()


#%% Box elec price

def plot_elec_price_box():
    df = df_nodal_el_price.copy()
    
    df.columns = [network.buses.loc[b].country for b in df_nodal_el_price.columns]
    df = df.iloc[:,df.columns != '']
    df = df.groupby(df.columns,axis=1).mean()

    # Rearange collumns to match index order 
    df = df[index_order]
    df['year'] = df_chain['year']

    optimal_index = df_chain.query('year == "scenarios"').index
    optimal_index = optimal_index[:-1]
    optimal_index = optimal_index.append(df_chain.query('s == 1 & c == 1').index)
    df_optimal = df.iloc[optimal_index]
    df_optimal['scenario'] = np.array(['local load','local 1990','Optimum',])

    # filter out burnin samples
    df = df[filt_burnin & filt_co2_cap]
    df.reindex()

    plot_box(df,df_optimal,ylabel='€/MWh',
                           title='Elec price',
                           save=True,
                           prefix=prefix,
                           name='elec_price',
                           ylim=(0,100),
                           fliersize=0.5,
                           linewidth=1,
                           color='tab:blue'
                           )

plot_elec_price_box()
#%% Box Co2 price

def plot_co2_price_box():
    df = df_nodal_co2_price.copy()

    df = df[index_order]
    df['year'] = df_chain['year']


    optimal_index = df_chain.query('year == "scenarios"').index
    optimal_index = optimal_index[:-1]
    optimal_index = optimal_index.append(df_chain.query('s == 1 & c == 1').index)
    df_optimal = df.iloc[optimal_index]
    df_optimal['scenario'] = np.array(['local load','local 1990','Optimum',])


    # filter out burnin samples
    df = df[filt_burnin & filt_co2_cap]

    plot_box(df,df_optimal,ylabel='€/T',
                           title='CO2 policy price',
                           save=True,
                           prefix=prefix,
                           name='co2_price',
                           ylim=(-2,100),
                           fliersize=0.5,
                           linewidth=1,
                           color='tab:blue'
                           )

plot_co2_price_box()

#%%

def plot_allocated_co2():

    df = df_theta.copy()
    df.columns=mcmc_variables


    for year in networks:
        co2_budget = networks[year].global_constraints.loc['CO2Limit'].constant
        df.loc[df_chain.year == year] = df.loc[df_chain.year == year].multiply(co2_budget)

    df = df.iloc[:,:33] #- df_co2.iloc[:,:33]

    df = df[df.mean().sort_values().index]
    df['year'] = df_chain['year']
    df = df[filt_burnin & filt_co2_cap]

    #df['year'] = df_chain['year']
    plot_box(df,title='Allocated Co2',prefix=prefix,name='allocated_co2',save=True)

plot_allocated_co2()


#%%

def plot_unused_co2():

    df = df_co2_assigned.copy()
    df = df_co2/df

    df = (df[df.mean().sort_values(ascending=False).index])*100
    df = df.iloc[:,:33]
    df['year'] = df_chain['year']
    #df['year'] = df_chain['year']

    optimal_index = df_chain.query('year == "scenarios"').index
    optimal_index = optimal_index[:-1]
    optimal_index = optimal_index.append(df_chain.query('s == 1 & c == 1').index)
    df_optimal = df.iloc[optimal_index]
    df_optimal['scenario'] = np.array(['local load','local 1990','Optimum',])

    # filter out burnin samples
    df = df[filt_burnin & filt_co2_cap]

    index_order = list(df.mean().sort_values(ascending=False).index)
    index_order.append('year')

    df = df[index_order]

    plot_box(df,df_optimal,title='Unused Co2',
                ylabel='% CO2 quotas used',
                prefix=prefix,
                name='unused_co2',
                save=True,
                fliersize=0.5,
                linewidth=1,
                color='tab:blue')



plot_unused_co2()

#%% Correlation of CO2 allocations


def plot_co2_correlation_matrix():

    f, axes = plt.subplots( figsize=(15, 11), sharex=True, sharey=False)

    corr = df_co2[df_chain.s>burnin_samples].corr()
    sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                square=True,
                #ax = ax
                )
    plt.savefig(f'graphics/co2_correlations_{prefix}.jpeg')

plot_co2_correlation_matrix()


#%% Plot co2 allocated vs el price 

#f,ax = plt.subplots(1,3)

def plot_country_co2_vs_elec_price(countries):
    
    f,ax = plt.subplots(1,5,sharey=True,figsize=(12,3))

    for i,country in enumerate(countries):

        country_idx = np.where(np.array(mcmc_variables)==country)[0]
        #plt.scatter((df_theta*base_emis).iloc[:,country_idx],df_country_el_price[country])
        #
        # ax[i].scatter((df_theta*base_emis*0.45).iloc[:,country_idx],df_co2[country][:8011],alpha=0.2)

        x = df_co2_assigned[country]
        x = x[filt_burnin & filt_co2_cap]

        y = (df_co2[country].iloc[:-3]/df_co2_assigned[country])*100
        y = y[filt_burnin & filt_co2_cap]

        ax[i].scatter(x,y,alpha=0.01)
        ax[i].set_xticks(np.arange(0, max(x)+1, 2e7))
        
        ax[i].set_xlabel('Assigned\nCO2 quotas')
        ax[i].set_title(country)
    ax[0].set_ylabel('% CO2 quotas used')
    plt.savefig(f'graphics/co2_vs_co2_{countries}.jpeg',dpi=400,bbox_inches='tight')

plot_country_co2_vs_elec_price(['PL','NL','AT','FI','SE'])


#%% Plot of brownfield capacities
import matplotlib
from matplotlib.patches import Circle, Ellipse
from matplotlib.legend_handler import HandlerPatch
import cartopy.crs as ccrs

tech_colors = {
    "onwind" : "b"                     ,
    "onshore wind" : "b",
    'offwind' : "c",
    'offshore wind' : "c",
    'offwind-ac' : "c",
    'offshore wind (AC)' : "c",
    'offwind-dc' : "#009999",
    'offshore wind (DC)' : "#009999",
    'wave' : "#004444",
    "hydro" : "#3B5323",
    "hydro reservoir" : "#3B5323",
    "ror" : "#78AB46",
    "run of river" : "#78AB46",
    'hydroelectricity' : '#006400',
    'solar' : "y",
    'solar PV' : "y",
    'solar thermal' : 'coral',
    'solar rooftop' : '#e6b800',
    "OCGT" : "wheat",
    "OCGT marginal" : "sandybrown",
    "OCGT-heat" : "orange",
    "gas boiler" : "orange",
    "gas boilers" : "orange",
    "gas boiler marginal" : "orange",
    "gas-to-power/heat" : "orange",
    "gas" : "brown",
    "natural gas" : "brown",
    "SMR" : "#4F4F2F",
    "oil" : "#B5A642",
    "oil boiler" : "#B5A677",
    "lines" : "k",
    "transmission lines" : "k",
    "H2" : "m",
    "hydrogen storage" : "m",
    "battery" : "slategray",
    "battery storage" : "slategray",
    "home battery" : "#614700",
    "home battery storage" : "#614700",
    "Nuclear" : "r",
    "Nuclear marginal" : "r",
    "nuclear" : "r",
    "uranium" : "r",
    "Coal" : "k",
    "coal" : "k",
    "Coal marginal" : "k",
    "Lignite" : "grey",
    "lignite" : "grey",
    "Lignite marginal" : "grey",
    "CCGT" : "orange",
    "CCGT marginal" : "orange",
    "heat pumps" : "#76EE00",
    "heat pump" : "#76EE00",
    "air heat pump" : "#76EE00",
    "ground heat pump" : "#40AA00",
    "power-to-heat" : "#40AA00",
    "resistive heater" : "pink",
    "Sabatier" : "#FF1493",
    "methanation" : "#FF1493",
    "power-to-gas" : "#FF1493",
    "power-to-liquid" : "#FFAAE9",
    "helmeth" : "#7D0552",
    "helmeth" : "#7D0552",
    "DAC" : "#E74C3C",
    "co2 stored" : "#123456",
    "CO2 sequestration" : "#123456",
    "CC" : "k",
    "co2" : "#123456",
    "co2 vent" : "#654321",
    "solid biomass for industry co2 from atmosphere" : "#654321",
    "solid biomass for industry co2 to stored": "#654321",
    "gas for industry co2 to atmosphere": "#654321",
    "gas for industry co2 to stored": "#654321",
    "Fischer-Tropsch" : "#44DD33",
    "kerosene for aviation": "#44BB11",
    "naphtha for industry" : "#44FF55",
    "land transport oil" : "#44DD33",
    "water tanks" : "#BBBBBB",
    "hot water storage" : "#BBBBBB",
    "hot water charging" : "#BBBBBB",
    "hot water discharging" : "#999999",
    "CHP" : "r",
    "CHP heat" : "r",
    "CHP electric" : "r",
    "PHS" : "g",
    "Ambient" : "k",
    "Electric load" : "b",
    "Heat load" : "r",
    "heat" : "darkred",
    "rural heat" : "#880000",
    "central heat" : "#b22222",
    "decentral heat" : "#800000",
    "low-temperature heat for industry" : "#991111",
    "process heat" : "#FF3333",
    "heat demand" : "darkred",
    "electric demand" : "k",
    "Li ion" : "grey",
    "district heating" : "#CC4E5C",
    "retrofitting" : "purple",
    "building retrofitting" : "purple",
    "BEV charger" : "grey",
    "V2G" : "grey",
    "land transport EV" : "grey",
    "electricity" : "k",
    "gas for industry" : "#333333",
    "solid biomass for industry" : "#555555",
    "industry electricity" : "#222222",
    "industry new electricity" : "#222222",
    "process emissions to stored" : "#444444",
    "process emissions to atmosphere" : "#888888",
    "process emissions" : "#222222",
    "oil emissions" : "#666666",
    "land transport fuel cell" : "#AAAAAA",
    "biogas" : "#800000",
    "solid biomass" : "#DAA520",
    "today" : "#D2691E",
    "shipping" : "#6495ED",
    "electricity distribution grid" : "#333333"}

bus_size_factor = 80000
linewidth_factor = 2000

# Get pie chart sizes for technology capacities 
tech_types =  list(network.generators.query('p_nom_extendable == False').carrier.unique()) + list(network.links.query('p_nom_extendable == False').carrier.unique())
tech_types.remove('DC')

bus_cap = pd.Series()
bus_cap.index = pd.MultiIndex.from_arrays([[],[]],names=['bus','tech'])
for tech in tech_types:
    s = network.generators.query(f'carrier == "{tech}" & p_nom_extendable == False').p_nom_opt.groupby(network.generators.bus).sum()

    if len(s)<=1:
        s = network.links.query(f'carrier == "{tech}" & p_nom_extendable == False').p_nom_opt.groupby(network.links.bus1).sum()

    s.index = pd.MultiIndex.from_arrays([s.index,[tech]*len(s)],names=['bus','tech'])
    bus_cap = pd.concat([bus_cap,s])

network_buses = network.buses.query('country != ""').index
bus_cap = bus_cap[bus_cap.index.get_level_values(0).isin(network_buses)]

# DC Link witdhts 
link_width = pd.Series(index=network.links.index)
link_width[network.links.query('carrier == "DC"').index] = network.links.query('carrier == "DC"').p_nom_opt
link_width[network.links.query('carrier != "DC"').index] = 0

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
fig.set_size_inches(7, 7)

#tech_colors = {'offwind':'#1f77b4','solar':'#ff7f0e','onwind':'#8c564b','CCGT':'#e377c2','OCGT':'#d62728','nuclear':'#d62728','ror':'#d62728'}

network.plot(
        bus_sizes=bus_cap/bus_size_factor,
        bus_colors=tech_colors,
        #line_colors=ac_color,
        link_colors='blue',
        line_widths=network.lines.s_nom / linewidth_factor,
        line_colors='#2ca02c',
        link_widths=link_width/linewidth_factor,
        #ax=ax[int(np.floor(i/2)),i%2],  
        boundaries=(-10, 30, 34, 70),
        color_geomap={'ocean': 'white', 'land': (203/255, 203/255, 203/255)})

#ax[int(np.floor(i/2)),i%2].set_title(plot_titles[i],font=font)


def make_legend_circles_for(sizes, scale=1.0, **kw):
    return [Circle((0, 0), radius=(s / scale)**0.5, **kw) for s in sizes]

def make_handler_map_to_scale_circles_as_in(ax, dont_resize_actively=False):
    fig = ax.get_figure()

    def axes2pt():
        return np.diff(ax.transData.transform([(0, 0), (1, 1)]), axis=0)[
            0] * (72. / fig.dpi)

    ellipses = []
    if not dont_resize_actively:
        def update_width_height(event):
            dist = axes2pt()
            for e, radius in ellipses:
                e.width, e.height = 2. * radius * dist
        fig.canvas.mpl_connect('resize_event', update_width_height)
        ax.callbacks.connect('xlim_changed', update_width_height)
        ax.callbacks.connect('ylim_changed', update_width_height)

    def legend_circle_handler(legend, orig_handle, xdescent, ydescent,
                              width, height, fontsize):
        w, h = 2. * orig_handle.get_radius() * axes2pt()
        e = Ellipse(xy=(0.5 * width - 0.5 * xdescent, 0.5 *
                        height - 0.5 * ydescent), width=w, height=w)
        ellipses.append((e, orig_handle.get_radius()))
        return e
    return {Circle: HandlerPatch(patch_func=legend_circle_handler)}

# Legend for bus size
handles = make_legend_circles_for(
    [3e7, 1e7], scale=bus_size_factor, facecolor="gray")

labels = ["  {} GW".format(s) for s in (300, 100)]
l2 = ax.legend(handles, labels,
                loc="upper left", bbox_to_anchor=(1.01, 1.4),
                labelspacing=3.0,
                framealpha=1.,
                title='Installed capacity',
                handler_map=make_handler_map_to_scale_circles_as_in(ax))
ax.add_artist(l2)

# Legend for carriers 
handles = []
for t in tech_types:
    s = 5e6
    scale = bus_size_factor,
    kw = {'facecolor':tech_colors[t]}
    handles.append(Circle((0, 0), radius=(s / bus_size_factor)**0.5, **kw))

labels = ["{}".format(s) for s in tech_types]
l1 = ax.legend(handles, labels,
                loc="upper left", bbox_to_anchor=(1.01, 0.85),
                labelspacing=1.0,
                framealpha=1.,
                title='Carriers',
                handler_map=make_handler_map_to_scale_circles_as_in(ax))
ax.add_artist(l1)

plt.savefig(f'graphics/brownfield_tech{prefix}.jpeg',dpi=fig.dpi,bbox_extra_artists=(l2,l1),bbox_inches='tight')



#%% Geographical Potentials 

bus_size_factor = 80000
linewidth_factor = 2000
# Get pie chart sizes for technology capacities 
tech_types =  list(network.generators.query('p_nom_max < 1e9').carrier.unique())
#tech_types.remove('DC')

bus_cap = pd.Series()
bus_cap.index = pd.MultiIndex.from_arrays([[],[]],names=['bus','tech'])
for tech in tech_types:
    s = (network.generators_t.p_max_pu[network.generators.query(f'carrier == "{tech}" & p_nom_extendable == True').index].mean() * network.generators.query(f'carrier == "{tech}" & p_nom_extendable == True').p_nom_max).groupby(network.generators.bus).sum()

    if len(s)<=1:
        s = network.links.query(f'carrier == "{tech}" & p_nom_extendable == True').p_nom_max.groupby(network.links.bus1).sum()


    s.index = pd.MultiIndex.from_arrays([s.index,[tech]*len(s)],names=['bus','tech'])
    bus_cap = pd.concat([bus_cap,s])

network_buses = network.buses.query('country != ""').index
bus_cap = bus_cap[bus_cap.index.get_level_values(0).isin(network_buses)]


fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
fig.set_size_inches(7, 7)

network.plot(
        bus_sizes=bus_cap/bus_size_factor,
        bus_colors=tech_colors,
        #line_colors=ac_color,
        link_colors='blue',
        line_widths=network.lines.s_nom / linewidth_factor,
        line_colors='#2ca02c',
        link_widths=link_width/linewidth_factor,
        #ax=ax[int(np.floor(i/2)),i%2],  
        boundaries=(-10, 30, 34, 70),
        color_geomap={'ocean': 'white', 'land': (203/255, 203/255, 203/255)})



# def make_legend_circles_for(sizes, scale=1.0, **kw):
#     return [Circle((0, 0), radius=(s / scale)**0.5, **kw) for s in sizes]

# def make_handler_map_to_scale_circles_as_in(ax, dont_resize_actively=False):
#     fig = ax.get_figure()

#     def axes2pt():
#         return np.diff(ax.transData.transform([(0, 0), (1, 1)]), axis=0)[
#             0] * (72. / fig.dpi)

#     ellipses = []
#     if not dont_resize_actively:
#         def update_width_height(event):
#             dist = axes2pt()
#             for e, radius in ellipses:
#                 e.width, e.height = 2. * radius * dist
#         fig.canvas.mpl_connect('resize_event', update_width_height)
#         ax.callbacks.connect('xlim_changed', update_width_height)
#         ax.callbacks.connect('ylim_changed', update_width_height)

#     def legend_circle_handler(legend, orig_handle, xdescent, ydescent,
#                               width, height, fontsize):
#         w, h = 2. * orig_handle.get_radius() * axes2pt()
#         e = Ellipse(xy=(0.5 * width - 0.5 * xdescent, 0.5 *
#                         height - 0.5 * ydescent), width=w, height=w)
#         ellipses.append((e, orig_handle.get_radius()))
#         return e
#     return {Circle: HandlerPatch(patch_func=legend_circle_handler)}

# Legend for bus size
handles = make_legend_circles_for(
    [3e7, 1e7], scale=bus_size_factor, facecolor="gray")

labels = ["  {} GW".format(s) for s in (300, 100)]
l2 = ax.legend(handles, labels,
                loc="upper left", bbox_to_anchor=(1.01, 1.4),
                labelspacing=3.0,
                framealpha=1.,
                title='Geographical potential',
                handler_map=make_handler_map_to_scale_circles_as_in(ax))
ax.add_artist(l2)

# Legend for carriers 
handles = []
for t in bus_cap.index.get_level_values(1).unique():
    s = 5e6
    scale = bus_size_factor,
    kw = {'facecolor':tech_colors[t]}
    handles.append(Circle((0, 0), radius=(s / bus_size_factor)**0.5, **kw))

labels = ["{}".format(s) for s in tech_types]
l1 = ax.legend(handles, labels,
                loc="upper left", bbox_to_anchor=(1.01, 0.85),
                labelspacing=1.0,
                framealpha=1.,
                title='Carriers',
                handler_map=make_handler_map_to_scale_circles_as_in(ax))
ax.add_artist(l1)

plt.savefig(f'graphics/geographic_potentials_{prefix}.jpeg',dpi=fig.dpi,bbox_extra_artists=(l2,l1),bbox_inches='tight')

#%%
import matplotlib.cm as cm
from matplotlib.colors import Normalize 
import matplotlib.artist as artist

bus_size_factor = 1
linewidth_factor = 2000
# Get pie chart sizes for technology capacities 

m_index = [(bus,bus+' '+ tech) for tech in ['onwind-2030','solar-2030'] for bus in network.buses.query('carrier == "AC"').index]
m_index = pd.MultiIndex.from_tuples(m_index)
bus_cap = pd.Series(index=m_index,data=0.5*np.ones(len(m_index)))


network_buses = network.buses.query('country != ""').index
bus_cap = bus_cap[bus_cap.index.get_level_values(0).isin(network_buses)]


cmap_wind = cm.PuBu
cmap_solar = cm.YlOrBr
norm_wind = Normalize(vmin=0,vmax=network.generators_t.p_max_pu[network.generators.query('carrier == "onwind"').index].mean().max())
norm_solar = Normalize(vmin=0,vmax=network.generators_t.p_max_pu[network.generators.query('carrier == "solar"').index].mean().max())

solar_col = cmap_solar(norm_solar(network.generators_t.p_max_pu[network.generators.query('carrier == "solar"').index].mean()))
solar_col = pd.Series(data=map(tuple,solar_col),index=network.generators.query('carrier == "solar"').index)

wind_col = cmap_wind(norm_wind(network.generators_t.p_max_pu[network.generators.query('carrier == "onwind"').index].mean()))
wind_col = pd.Series(data=map(tuple,wind_col),index=network.generators.query('carrier == "onwind"').index)

bus_color = pd.concat((wind_col,solar_col))

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
fig.set_size_inches(8.5, 7)

network.plot(
        bus_sizes=bus_cap/bus_size_factor,
        bus_colors=bus_color,
        #line_colors=ac_color,
        link_colors='blue',
        line_widths=network.lines.s_nom *0,#/ linewidth_factor,
        line_colors='#2ca02c',
        link_widths=link_width *0,#/linewidth_factor,
        #ax=ax[int(np.floor(i/2)),i%2],  
        boundaries=(-10, 30, 34, 70),
        color_geomap={'ocean': 'white', 'land': (203/255, 203/255, 203/255)})


data = np.linspace(0,norm_wind.vmax, 100).reshape(10, 10)
cax = fig.add_axes([0.87, 0.27, 0.05, 0.5])
im = ax.imshow(data, cmap=cmap_wind,visible=False)
fig.colorbar(im, cax=cax, orientation='vertical')

data = np.linspace(0,norm_solar.vmax, 100).reshape(10, 10)
cax = fig.add_axes([0.97, 0.27, 0.05, 0.5])
im = ax.imshow(data, cmap=cmap_solar,visible=False)
fig.colorbar(im, cax=cax, orientation='vertical')


plt.savefig(f'graphics/capacit_factor_{prefix}.jpeg',dpi=fig.dpi,bbox_inches='tight')



#%% Abbility to cover demand with renewables

bus_size_factor = 200000
linewidth_factor = 2000
# Get pie chart sizes for technology capacities 
tech_types =  list(network.generators.query('p_nom_extendable == True').carrier.unique())
#tech_types.remove('DC')

bus_cap = pd.Series()
bus_cap.index = pd.MultiIndex.from_arrays([[],[]],names=['bus','tech'])
#for tech in tech_types:
    #try :
    #    s = network.generators.query(f'carrier == "{tech}" & p_nom_extendable == True').p_nom_max.groupby(network.generators.bus).sum()
    #except : 
    #    s= 0
    #if len(s)<=1:
    #    s = network.links.query(f'carrier == "{tech}" & p_nom_extendable == True').p_nom_max.groupby(network.links.bus1).sum()
#
corrected_capital = ((1/network.generators_t.p_max_pu.mean())*network.generators.capital_cost)
corrected_capital = corrected_capital[corrected_capital<1e9]
idxmin = corrected_capital.groupby(network.generators.bus).idxmin()

s = corrected_capital.groupby(network.generators.bus).min()

s.index = pd.MultiIndex.from_arrays([s.index,network.generators.loc[idxmin].carrier],names=['bus','tech'])
bus_cap = pd.concat([bus_cap,s])

network_buses = network.buses.query('country != ""').index
bus_cap = bus_cap[bus_cap.index.get_level_values(0).isin(network_buses)]

bus_cap = bus_cap-bus_cap.min()

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
fig.set_size_inches(7, 7)

cmap = plt.cm.rainbow
norm = matplotlib.colors.Normalize(vmin=1.5, vmax=4.5)
bus_col = cmap(norm(bus_cap.values))
bus_col_series= pd.Series(data=map(tuple,bus_col),index=bus_cap.index)

network.plot(
        bus_sizes=1e3,
        bus_colors=bus_col_series,
        #line_colors=ac_color,
        link_colors='blue',
        line_widths=network.lines.s_nom / linewidth_factor,
        line_colors='#2ca02c',
        link_widths=link_width/linewidth_factor,
        #ax=ax[int(np.floor(i/2)),i%2],  
        boundaries=(-10, 30, 34, 70),
        color_geomap={'ocean': 'white', 'land': (203/255, 203/255, 203/255)})



def make_legend_circles_for(sizes, scale=1.0, **kw):
    return [Circle((0, 0), radius=(s / scale)**0.5, **kw) for s in sizes]

def make_handler_map_to_scale_circles_as_in(ax, dont_resize_actively=False):
    fig = ax.get_figure()

    def axes2pt():
        return np.diff(ax.transData.transform([(0, 0), (1, 1)]), axis=0)[
            0] * (72. / fig.dpi)

    ellipses = []
    if not dont_resize_actively:
        def update_width_height(event):
            dist = axes2pt()
            for e, radius in ellipses:
                e.width, e.height = 2. * radius * dist
        fig.canvas.mpl_connect('resize_event', update_width_height)
        ax.callbacks.connect('xlim_changed', update_width_height)
        ax.callbacks.connect('ylim_changed', update_width_height)

    def legend_circle_handler(legend, orig_handle, xdescent, ydescent,
                              width, height, fontsize):
        w, h = 2. * orig_handle.get_radius() * axes2pt()
        e = Ellipse(xy=(0.5 * width - 0.5 * xdescent, 0.5 *
                        height - 0.5 * ydescent), width=w, height=w)
        ellipses.append((e, orig_handle.get_radius()))
        return e
    return {Circle: HandlerPatch(patch_func=legend_circle_handler)}

# Legend for bus size
handles = make_legend_circles_for(
    [3e7, 1e7], scale=bus_size_factor, facecolor="gray")

labels = ["  {} GW".format(s) for s in (300, 100)]
l2 = ax.legend(handles, labels,
                loc="upper left", bbox_to_anchor=(1.01, 1.4),
                labelspacing=3.0,
                framealpha=1.,
                title='Geographical potential',
                handler_map=make_handler_map_to_scale_circles_as_in(ax))
ax.add_artist(l2)

# Legend for carriers 
handles = []
for t in tech_types:
    s = 5e6
    scale = bus_size_factor,
    kw = {'facecolor':tech_colors[t]}
    handles.append(Circle((0, 0), radius=(s / bus_size_factor)**0.5, **kw))

labels = ["{}".format(s) for s in tech_types]
l1 = ax.legend(handles, labels,
                loc="upper left", bbox_to_anchor=(1.01, 0.85),
                labelspacing=1.0,
                framealpha=1.,
                title='Carriers',
                handler_map=make_handler_map_to_scale_circles_as_in(ax))
ax.add_artist(l1)



#%% New take on secondary metrics plot

metrics={'system cost':['system_cost'],
        'inequality energy production':['gini'],
        'inequality emission pr pop':['gini_co2_pr_pop'],
        'inequality system cost pr pop':['gini_cost_pop'],
        'energy transfer':['energy_dependance'],
        'inequality co2 pr mwh':['gini_co2_energy'],
        #'gini co2':['gini_co2'],
        #'autoarky':['autoarky']
        }

df = df_chain[['year']]
for key in metrics: 
    df[key] = df_secondary[metrics[key]].sum(axis=1)


#df = df_secondary[['system_cost','gini','gini_co2_pr_pop']]
#df['energy dependance'] = df_energy_dependance
#df['index'] = df.index

variables = df.columns[2:].values

f, axes = plt.subplots(len(variables), 1, figsize=(9, 15), sharex=True, sharey=False)

for ax,v in zip(axes.flat,variables):

    # Create a cubehelix colormap to use with kdeplot
    #cmap = sns.cubehelix_palette(start=s, light=1, as_cmap=True)

    # Generate and plot a random bivariate dataset
    #x, y = rs.normal(size=(2, 50))
    sns.histplot(
        x=df['system cost'], y=df[v],
        bins=25,
        #hue='year',
        #cmap=cmap, 
        #fill=True,
        #clip=(-5, 5), 
        #cut=10,
        #thresh=0, levels=15,
        ax=ax,
    )
    #ax.set_axis_off()

plt.savefig(f'graphics/secondary_metrics_{prefix}.jpeg')



#%%#########################################################################
################################# geo plot ########################################

def plot_geo(df,title='',colorbar=''):
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
                        colorbar_title = colorbar,
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
        width=500,
        height=500,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )


    fig.show()
    return fig


#%%

def plot_country_cost_co2(country='DK'):
    sns.histplot(x=df_secondary['system_cost'],y=df_co2[country])

    plt.savefig(f'graphics/country_co2_co2_{country}_{prefix}.jpeg')

plot_country_cost_co2(country='FI')

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

    #df = df_country_el_price.iloc[7750]
    #df = df_nodal_co2_price.iloc[36]
    #df = df_co2_pr_energy.iloc[27]
    df = (df_co2_sweep/(df_country_load*network.snapshot_weightings[0])*1000).iloc[0,:-1]

    fig = plot_geo(df,title='CO2 allocation',colorbar='kg CO2 pr MWh load')
    if save:
        fig.write_image(f'graphics/co2_gdp_geo_{prefix}.jpeg')

plot_co2_gpd_geo()
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

    df = df_theta.copy()
    df['index'] = df.index
    df['s'] = df_chain['s']
    df['c'] = df_chain['c']

    df = df.rename(columns=lambda s : 'value '+s if len(s)==2  else s)

    theta_long = pd.wide_to_long(df,['value ',],i=['index'],j='country',suffix='[A-Z][A-Z]')
    theta_long = theta_long.reset_index()

    #sns.set_theme(style="ticks")
    # Define the palette as a list to specify exact values
    #palette = sns.color_palette("rocket", as_cmap=True)

    
    #f, ax = plt.subplots(figsize=(10,30))
    # Plot the lines on two facets
    sns.relplot(
        data=theta_long.query('c <=1 '),
        x="s", y='value ',
        hue="country",
        palette='Set2',
        row='c',
        ci=None,
        kind="line",
        height=5, aspect=1.5,)

    plt.suptitle('chain development over time')
    if save:
        plt.savefig(f'graphics/chain_development_{prefix}.jpeg')
        #sns_plot.fig.show()

plot_chain_development(save=True)

#%% Plot acceptance rate over time 

def plot_acceptance_rate(save=True):
    x = df_chain.groupby(df_chain.s).mean().a
    N = 20 # Number of samples to average over
    move_avg = np.convolve(x, np.ones(N)/N, mode='valid')
    plt.plot(move_avg*100)
    plt.title('Accaptance rate')
    plt.ylabel('Average % accepted samples')
    plt.xlabel('Sample number')
    if save:
        plt.savefig(f'graphics/acceptance_{prefix}.jpeg')

plot_acceptance_rate()

#%% Plot of autocorrelation 


#fig,ax = plt.subplots()

def calc_autocorrelation(x):
    chain = x[0]
    theta = x[1]
    series = df_theta.iloc[df_chain.iloc[0:-3].query(f'c == {chain}').index][str(theta)]
    acf = np.correlate(series,series,mode='full')
    acf = acf[acf.size//2:]
    return acf

df_acf = pd.DataFrame(columns=['val','c','t','lag'])

for c in range(8):
    for t in range(30):
        acf = calc_autocorrelation((c,t))
        df = pd.DataFrame(acf,columns=['val'])
        df['lag'] = df.index
        df['c'] = [c]*df.shape[0]
        df['t'] = [t]*df.shape[0]
        df_acf = df_acf.append(df)
        

sns.lineplot(data =df_acf.sample(frac=0.2),x='lag',y='val',hue='t',style='c')



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
