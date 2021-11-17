import pandas as pd
import numpy as np 
from iso3166 import countries as iso_countries



def assign_locations(n):
    # Assigns locations to all components in the network 
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


def remove_duplicates(df):
    # Remove duplicates from dataframe
    index = df.index
    is_duplicate = index.duplicated(keep="first")
    not_duplicate = ~is_duplicate
    df = df[not_duplicate]
    return df

def create_networks_dataframes(networks):
    # df with generators and links from all networks
    for key in networks:
        networks[key] = assign_locations(networks[key])

    g = pd.concat([networks[n].generators for n in networks]).drop_duplicates()
    g = remove_duplicates(g)
    l = pd.concat([networks[n].links for n in networks]).drop_duplicates()
    l = remove_duplicates(l)

    s = pd.concat([networks[n].stores for n in networks]).drop_duplicates()
    s = remove_duplicates(s)

    su = pd.concat([networks[n].storage_units for n in networks]).drop_duplicates()
    su = remove_duplicates(su)

    return g, l, s, su

def calc_autoarky(dfs, gen, links, network):
    # Soverignity 
    bus_total_prod = dfs['df_gen_e'].groupby(gen.bus,axis=1).sum().groupby(network.buses.country,axis=1).sum()
    ac_buses = network.buses.query('carrier == "AC"').index
    generator_link_carriers = ['OCGT', 'CCGT', 'coal', 'lignite', 'nuclear', 'oil']
    filt = links.bus1.isin(ac_buses) & links.carrier.isin(generator_link_carriers)
    link_prod = dfs['df_links_E'][filt.index].loc[:,filt].groupby(links.location,axis=1).sum()
    link_prod[''] = 0

    bus_total_prod += link_prod
    bus_total_prod.pop('')
    bus_total_load = network.loads_t.p.sum().groupby(network.buses.country).sum()
    bus_prod_vs_load = bus_total_prod.divide(bus_total_load)
    bus_prod_vs_load['year'] = dfs['df_chain']['year']
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

def calc_co2_pr_gdp(dfs,network):
    model_countries = network.buses.country.unique()[:33]
    alpha3 = [iso_countries.get(c).alpha3 for c in model_countries]
    df_gdp_i = dfs['df_gdp'].set_index('Country Code')
    model_countries_gdp = pd.DataFrame(df_gdp_i.loc[alpha3]['2018'])
    model_countries_gdp.index = model_countries

    co2 = dfs['df_co2'].iloc[:,:33]
    co2_pr_gdp = co2.divide(model_countries_gdp['2018'],axis=1)

    return co2_pr_gdp

def calc_co2_pr_pop(dfs,network):
    model_countries = network.buses.country.unique()[:33]
    alpha3 = [iso_countries.get(c).alpha3 for c in model_countries]
    df_pop_i = dfs['df_pop'].set_index('Country Code')
    model_countries_pop = pd.DataFrame(df_pop_i.loc[alpha3]['2018'])
    model_countries_pop.index = model_countries

    co2 = dfs['df_co2'].iloc[:,:33]
    co2_pr_pop = co2.divide(model_countries_pop['2018'],axis=1)

    return co2_pr_pop


def calc_nodal_co2_reduction(dfs,network):
    # Nodal co2 reduction 
    #co2_totals_2 = pd.read_csv('data/elec_emission_incl_autoprod.csv',index_col=0)


    co2_totals = pd.read_csv('data/co2_totals.csv',index_col=0)
    co2_totals_elec = co2_totals['electricity']
    #co2_totals_elec = co2_totals.sum(axis=1)
    model_countries = network.buses.country.unique()
    co2_totals_elec = co2_totals_elec[model_countries[:-1]]

    dfs['df_nodal_co2_reduct'] = (dfs['df_co2'])/(co2_totals_elec*1e6)
    return

def update_secondary_data(dfs,network,base_emis,generators,links,scenario_names):
# Calc data for cost increase and co2 reduction 

    #optimal_index = dfs['df_chain'].query(f"year == '{scenario_names}'").index
    optimal_index = dfs['df_chain'].query('c==1 & s==1').index
    minimum_system_cost =  min(dfs['df_secondary'].loc[optimal_index,'system_cost'].values)

    #filt = dfs['df_chain'].query('year != "sweep_2030_f"').index
    #minimum_system_cost = min(dfs['df_secondary'].loc[filt,'system_cost'])
    #minimum_system_cost = network.objective_optimum

    #cost_increase = (dfs['df_secondary'].system_cost-network.objective_optimum)/network.objective_optimum*100
    cost_increase = ((dfs['df_secondary'].system_cost-minimum_system_cost)/minimum_system_cost)*100

    dfs['df_secondary']['cost_increase'] = cost_increase
    dfs['df_secondary']['co2_emission'] = dfs['df_co2'].sum(axis=1)
    dfs['df_secondary']['co2_reduction'] = 100 - dfs['df_co2'].sum(axis=1)/base_emis * 100


    autoarky = calc_autoarky(dfs, generators, links, network)
    dfs['df_secondary']['autoarky'] = autoarky

    co2_pr_gdp = calc_co2_pr_gdp(dfs,network)
    gini_co2_pr_gdp = calc_gini(co2_pr_gdp)
    dfs['df_secondary']['gini_co2_pr_gdp'] = gini_co2_pr_gdp

    co2_pr_pop = calc_co2_pr_pop(dfs,network)
    gini_co2_pr_pop = calc_gini(co2_pr_pop)
    dfs['df_secondary']['gini_co2_pr_pop'] = gini_co2_pr_pop

    

    return 

def create_tech_sum_df(dfs, networks,generators):
    # Dataset with with aggregated technology capacities

    links_carrier = pd.concat((networks[key].links.carrier for key in networks))
    links_carrier = links_carrier[~links_carrier.index.duplicated()]
    df_link_sum = dfs['df_links'].groupby(links_carrier,axis=1).sum()

    stores_carrier = pd.concat((networks[key].stores.carrier for key in networks))
    stores_carrier = stores_carrier[~stores_carrier.index.duplicated()]
    df_store_sum = dfs['df_store_P'].groupby(stores_carrier,axis=1).sum()
    df_store_sum.columns = [c + '_store' for c in df_store_sum.columns]
    #df_gen_sum = df_gen_p.groupby(network.generators.carrier,axis=1).sum()
    df_gen_sum = dfs['df_sum']
    #df_gen_sum.pop('oil')

    dfs['df_tech_sum'] = pd.concat([df_link_sum,df_gen_sum,df_store_sum],axis=1)

    # Dataset with aggregated technology energy production 
    dfs['df_link_e_sum'] = dfs['df_links_E'].groupby(links_carrier,axis=1).sum()

    dfs['df_store_e_sum'] = dfs['df_store_E'].groupby(stores_carrier,axis=1).sum()
    dfs['df_store_e_sum'].columns = [c + '_store' for c in dfs['df_store_e_sum'].columns]

    dfs['df_gen_e_sum'] = dfs['df_gen_e'].groupby(generators.carrier,axis=1).sum()

    dfs['df_tech_e_sum'] = pd.concat([dfs['df_link_e_sum'],dfs['df_gen_e_sum'],dfs['df_store_e_sum']],axis=1)

    return


def create_country_pop_df(dfs,network):
    model_countries = network.buses.country.unique()[:33]
    alpha3 = [iso_countries.get(c).alpha3 for c in model_countries]
    df_pop_i = dfs['df_pop'].set_index('Country Code')
    df_gdp_i = dfs['df_gdp'].set_index('Country Code')

    model_countries_pop = pd.DataFrame(df_pop_i.loc[alpha3]['2018'])
    model_countries_gdp = pd.DataFrame(df_gdp_i.loc[alpha3]['2018'])
    model_countries_pop.index = model_countries
    model_countries_gdp.index = model_countries
    
    dfs['df_country_pop'] = pd.Series(model_countries_pop['2018'])
    dfs['df_country_gdp'] = pd.Series(model_countries_gdp['2018'])
    return 


def calc_country_dfs(dfs,network,links,generators,storage_units):

    def set_multiindex(df,component):
        index = ((n,component.country[n],component.carrier[n]) for n in df.columns)
        m_index = pd.MultiIndex.from_tuples(index)
        df.columns = m_index

    set_multiindex(dfs['df_links_E'],links)
    set_multiindex(dfs['df_gen_e'],generators)
    set_multiindex(dfs['df_storage_E'],storage_units)
    #node_energy = dfs['df_links_E'].groupby(level=[1,2],axis=1).sum()

    # Filter any non electricity producting generators out of the dfs['df_gen_e'] dataframe 
    generator_el_energy = dfs['df_gen_e'].loc[:,(slice(None),dfs['df_gen_e'].columns.get_level_values(1) != '',slice(None))]

    energy_generating_links = ['OCGT','H2 Fuel Cell','battery discharger','home battery discharger','CCGT','coal','lignite','nuclear','oil']
    energy_consuming_links = ['H2 Electrolysis','battery charger','Sabatier','helmeth','home battery charger']
    energy_distributing_links = ['DC','H2 pipeline','electricity distribution grid'] 

    # Multiply generating links with their efficiency 
    link_generators_energy = dfs['df_links_E'].loc[:,(slice(None),slice(None),energy_generating_links)] 
    eff = links.loc[link_generators_energy.columns.get_level_values(0)].efficiency.values
    link_generators_energy = link_generators_energy*eff
    link_consumors_energy = - dfs['df_links_E'].loc[:,(slice(None),slice(None),energy_consuming_links)] 

    dfs['df_energy'] = pd.concat((link_consumors_energy,link_generators_energy,dfs['df_storage_E'],generator_el_energy),axis=1)

    dfs['df_country_energy'] = dfs['df_energy'].groupby(level=[1],axis=1).sum()

    dfs['df_country_load'] = network.loads_t.p.sum().groupby(network.buses.country).sum()

    dfs['df_country_k'] = dfs['df_country_energy']/dfs['df_country_load']

    dfs['df_country_export'] = dfs['df_country_energy']-dfs['df_country_load']

    dfs['df_energy_dependance'] =  dfs['df_country_export'][dfs['df_country_export']>0].sum(axis=1)

    dfs['df_country_cost'] = dfs['df_nodal_cost'].groupby(level=[2],axis=1).sum().groupby(network.buses.country,axis=1).sum()
    dfs['df_country_cost'] = dfs['df_country_cost'].iloc[:,1:]

    dfs['df_nodal_cost_marginal'] = dfs['df_nodal_cost'].loc[:,(slice(None),'marginal',slice(None))]
    dfs['df_country_marginal_cost'] = dfs['df_nodal_cost_marginal'].groupby(level=[2],axis=1).sum().groupby(network.buses.country,axis=1).sum()
    dfs['df_country_marginal_cost'] = dfs['df_country_marginal_cost'].iloc[:,1:]

    return



def data_postprocess(dfs,networks,base_emis,co2_red=0.45,scenario_names='scenarios_2030_f'):
    
    network = list(networks.values())[0]

    generators, links, stores, storage_units = create_networks_dataframes(networks)
    update_secondary_data(dfs,network,base_emis,generators,links,scenario_names)


    create_tech_sum_df(dfs, networks,generators)

    create_country_pop_df(dfs,network)

    calc_country_dfs(dfs,network,links,generators,storage_units)

    dfs['df_secondary']['gini_cost_pop'] = calc_gini(dfs['df_country_cost']/dfs['df_country_pop'])
    dfs['df_secondary']['gini_cost_energy'] = calc_gini(dfs['df_country_cost']/dfs['df_country_pop'])
    dfs['df_secondary']['gini_co2_energy'] = calc_gini(dfs['df_country_cost']/dfs['df_country_pop'])
    dfs['df_secondary']['gini_cost'] = calc_gini(dfs['df_country_cost'])
    dfs['df_secondary']['gini_marginal_cost'] = calc_gini(dfs['df_country_marginal_cost'])
    dfs['df_secondary']['energy_dependance'] = dfs['df_energy_dependance']
    dfs['df_secondary']['gini_co2_price'] = calc_gini(dfs['df_nodal_co2_price']/dfs['df_country_pop'])

    df_country_el_price = dfs['df_nodal_el_price'][network.buses.query('carrier == "AC"').index].groupby(network.buses.country,axis=1).sum()
    dfs['df_secondary']['gini_el_price'] = calc_gini(df_country_el_price/dfs['df_country_pop'])

    dfs['df_theta'].columns = dfs['mcmc_variables']
    dfs['df_co2_assigned'] = dfs['df_theta']*base_emis*co2_red

    calc_nodal_co2_reduction(dfs,network)

    return 

