#%%

import pypsa
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib
#import matplotlib
#matplotlib.interactive(False)
import seaborn as sns
from _mcmc_helpers import *
from _helpers import make_override_component_attrs, get_tech_colors
from data_process import data_postprocess
import plotly.graph_objects as go
import matplotlib
from matplotlib.patches import Circle, Ellipse
from matplotlib.legend_handler import HandlerPatch
import cartopy.crs as ccrs

#%load_ext autoreload
#%autoreload 2

if os.path.split(os.getcwd())[1] == 'scripts':
    os.chdir('..')

#%%#################### import datasets ####################################
######################## import datasets ###################################

prefix = '2030_elec_f'
sector = 'e'
burnin_samples = 100

emis_1990 =   1481895952.8
emis_1990_H = 2206933437.8055553

base_emis = emis_1990

datasets = ['mcmc_2030_f', 'sweep_2030_f', 'scenarios_2030_f']

scenarios = ['Grandfathering', 'Sovrignity', 'Efficiency', 'Egalitarianism', 'Ability to pay']


def import_datasets(datasets):
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

    dfs = {}
    networks = {}
    override_component_attrs = make_override_component_attrs()
    for run_name in datasets:
        networks[run_name] = pypsa.Network(f'results/{run_name}/network_c0_s1.nc',
                                            override_component_attrs=override_component_attrs)

        for df_name in df_names.keys():
            if df_name =='df_nodal_cost':
                df = pd.read_csv(f'results/{run_name}/'+df_names[df_name],index_col=0,header=[0,1,2,3])
            else : 
                df = pd.read_csv(f'results/{run_name}/'+df_names[df_name],index_col=0)
            if df_name == 'df_chain':
                df['year'] = run_name
            try :
                dfs[df_name] = pd.concat((dfs[df_name],df),ignore_index=True)
                #vars()[df_name] = dfs[df_name]
            except Exception:
                dfs[df_name] = df
                #vars()[df_name] = dfs[df_name]

    #network = networks[datasets[0]]
    dfs['mcmc_variables'] = read_csv(networks[run_name].mcmc_variables)
    dfs['mcmc_variables'] = [row[0]+row[1] for row in dfs['mcmc_variables']]

    # GDP and Population data '
    dfs['df_pop'] = pd.read_csv('data/API_SP.POP.TOTL_DS2_en_csv_v2_2106202.csv',
                        sep=',',
                        index_col=0,skiprows=3)
    dfs['df_gdp'] = pd.read_csv('data/API_NY.GDP.MKTP.CD_DS2_en_csv_v2_2055594.csv',
                            sep=',',
                            index_col=0,skiprows=3)
    
    return dfs,networks 


dfs,networks = import_datasets(datasets)
network = list(networks.values())[0]

#%%##### Data postprocessing #################
######### Data postprocessing ################

data_postprocess(dfs,networks,base_emis,co2_red=0.45)

# Filters 
filt_co2_cap = dfs['df_co2'].sum(axis=1)<=base_emis*0.448
filt_burnin = dfs['df_chain'].s>burnin_samples



# Correction of wrong optimal solution 
#i_optimum = 31232

i_sweep = dfs['df_chain'].query('year == "sweep_2030_f"').index
i_optimum = dfs['df_secondary'].iloc[i_sweep].query('co2_reduction <= 55 & co2_reduction >= 54.999').index 

i_s1 = dfs['df_chain'].query('s==1').index
for df in dfs:
    if df not in ['mcmc_variables','df_country_pop','df_gdp','df_pop','df_country_gdp','df_country_load','df_chain']:
        dfs[df].iloc[i_s1] = dfs[df].iloc[i_optimum]


#%%############# plots #############################
# ############################################
# ###########################################
# ####### Corrolelogram cost vs co2 #########

def plot_cost_vs_co2(prefix='',save=False,
                    title= 'Cost vs emission reduction',
                    plot_sweep=False,
                    plot_optimum=False,
                    plot_scenarios=False,
                    sweep_name='sweep_2030_f',
                    co2_emis_level = 55):
    
    # Create dataframe with relevant data
    df = dfs['df_secondary'][['cost_increase']]
    #df['co2 emission'] = df_co2.sum(axis=1)
    df['co2 emission'] =dfs['df_store_P']['co2 atmosphere']
    df['co2 reduction'] = 1-(df['co2 emission']/base_emis )
    df['co2 reduction'] = df['co2 reduction']*100
    df['year'] = dfs['df_chain']['year']

    co2_label = 'CO2 reduction [%]'
    cost_label = 'Cost increase [%]'

    df.rename(columns={'cost_increase':cost_label,'co2 reduction':co2_label},inplace=True)
    #df = df[dfs['df_chain'].a==1]

    # Create dataframe with optimal solution
    if plot_optimum:
        index_optimum = dfs['df_chain'].query('c==1 & s==1').index
        #index_optimum = dfs['df_chain'].query('year == "scenarios_2030_f"').index
        df_optimum = df.iloc[index_optimum]

    if plot_sweep:
        index_sweep = dfs['df_chain'].query(f'year == "{sweep_name}"').index
        df_sweep = df.iloc[index_sweep]

    if plot_sweep:
        index_scenarios = dfs['df_chain'].query(f'year == "scenarios_2030_f"').index
        df_scenarios = df.iloc[index_scenarios]

    # filter out burnin samples
    #df = df[ filt_burnin & filt_co2_cap]
    df = df[ filt_burnin ]

    cost_limits = [df[cost_label].min(),df[cost_label].max()]
    co2_limits = [df[co2_label].min(),df[co2_label].max()]

    def sweep_plot(xdata,ydata,**kwargs):
        plt.gca()
        styles = ['bD','ms']
        #for i in range(2):
        plt.plot(df_sweep[co2_label],
                df_sweep[cost_label],
                        #styles[i],
                        #marker='D',
                        #mfc='g',
                        markersize=10)

    # Function for plotting vertical lines for co2 limit 
    def plot_lower(xdata,ydata,**kwargs):
        plt.gca().vlines(co2_emis_level,
                        ymin=cost_limits[0],
                        ymax=cost_limits[1],
                        colors = 'r',
                        #colors=kwargs['color']
                        )

    def optimum_plot(xdata,ydata,**kwargs):
        plt.gca()
        plt.plot(df_optimum[co2_label],
                df_optimum[cost_label],
                        marker='X',
                        mfc='r',
                        markersize=20)

    def scenarios_plot(xdata,ydata,**kwargs):
        scenario_names = scenarios#['Local Load','Local 1990','Optimum','EU ETS']
        x = df_scenarios[co2_label]
        y = df_scenarios[cost_label]
        plt.gca()
        plt.scatter(x,
                    y,
                    c = '#f58905',
                    s = 50,
                        )
        for i,txt in enumerate(scenario_names):
            if txt == 'Egalitarianism' :
                offset = 0.5
            elif txt == 'Ability to pay' :
                offset = -0.9
            else :
                offset = 0 
            plt.gca().annotate(txt, (x.iloc[i], y.iloc[i]+offset),fontsize=14)



    sns_plot = sns.pairplot(df, 
                            vars=[co2_label,cost_label],
                            kind="hist", 
                            diag_kind='hist',
                            #hue='year',
                            plot_kws=dict(bins=50,thresh=0),
                            diag_kws=dict(bins=40,kde=False,log_scale=False),
                            aspect=1.6,
                            height=3,
                            palette='Set2')
    #plt.suptitle(title)

    cost_limits = [df[cost_label].min(),df[cost_label].max()]
    sns_plot.map_lower(plot_lower)

    if plot_sweep:
        sns_plot.map_lower(sweep_plot)

    # Draw optimal solution on plot 
    if plot_optimum:
        sns_plot.map_lower(optimum_plot)

    # Draw optimal solution on plot 
    if plot_scenarios:
        sns_plot.map_lower(scenarios_plot)

    #sns_plot.axes[0,0].set_ylim((0.45,1))

    if save:
        sns_plot.savefig(f'graphics/cost_vs_co2_{prefix}.jpeg')


plot_cost_vs_co2(save=True,prefix=prefix,plot_sweep=True,plot_optimum=True,plot_scenarios=True)

#%%############ Plot Boxplots ############################

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
        df_optimal = pd.melt(df_wide_optimal,value_vars=model_countries,id_vars=['year','Scenario'],var_name='Country')
        sns.stripplot(x='Country',y='value',#hue='Country',
                        data=df_optimal,
                        order=df_wide.columns[:-1],
                        #jitter=0.5,
                        hue='Scenario',
                        linewidth=1,
                        #marker='^',
                        #palette={'local load':'g','local 1990':'orange','Optimum unconstrained':'c','Optimum':'r','EU ETS':'b'},
                        palette='Dark2',
                        size=5,
                        ax=ax,)


    plt.ylabel(ylabel)
    plt.gca().set_title(title, y=1.0, pad=-14)
    #plt.suptitle(title)

    if ylim != None:
        plt.ylim(ylim)

    if save:
        plt.savefig(f'graphics/{name}_{prefix}.jpeg',transparent=False,dpi=400)

#%% Set index order for box plots

index_order = (dfs['df_co2']/dfs['df_country_energy']).mean().sort_values().index

if not 'year' in index_order:
    index_order = index_order.append(pd.Index(['year']))

#%%

def plot_co2_pr_mwh_box():

    df = dfs['df_co2']/dfs['df_country_energy']
    df['year'] = dfs['df_chain']['year']

    # Rearange collumns to match index order 
    df = df[index_order]
    df['year'] = dfs['df_chain']['year']

    optimal_index = dfs['df_chain'].query('year == "scenarios_2030_f"').index
    #optimal_index = optimal_index[:-1]
    #optimal_index = optimal_index.append(dfs['df_chain'].query('s == 1 & c == 1').index)
    df_optimal = df.iloc[optimal_index]
    #df_optimal['scenario'] = np.array(['local load','local 1990','Optimum uc','EU ETS','Optimum',])
    df_optimal['Scenario'] = np.array(scenarios)
    #df_optimal = df_optimal.iloc[[0,1,3,4]]

    # filter out burnin samples
    df = df[filt_burnin & filt_co2_cap]

    plot_box(df,df_optimal,ylabel='T Co2 pr MWh produced',
                           title='Co2 intensity pr MWh',
                           save=True,
                           prefix=prefix,
                           name='co2_mwh_box',
                           fliersize=0.1,
                           linewidth=0.5,
                           color='#8f897b'
                           )
    plt.gca()

#plt.ylim((0,5e4))
plot_co2_pr_mwh_box()


#%% Box elec price

def plot_elec_price_box():
    df = dfs['df_nodal_el_price'].copy()
    
    df.columns = [network.buses.loc[b].country for b in dfs['df_nodal_el_price'].columns]
    df = df.iloc[:,df.columns != '']
    df = df.groupby(df.columns,axis=1).mean()

    df = df[df.median().sort_values().index]
    # Rearange collumns to match index order 
    df['year'] = dfs['df_chain']['year']
    #df = df[index_order]
    
    

    optimal_index = dfs['df_chain'].query('year == "scenarios_2030_f"').index
    #optimal_index = optimal_index[:-1]
    #optimal_index = optimal_index.append(dfs['df_chain'].query('s == 1 & c == 1').index)
    df_optimal = df.iloc[optimal_index]
    df_optimal['Scenario'] = np.array(scenarios)

    #df_optimal = df_optimal.iloc[[0,1,3,4]]
    # filter out burnin samples
    df = df[filt_burnin & filt_co2_cap]
    df.reindex()

    plot_box(df,df_optimal,ylabel='€/MWh',
                           title='Electricity price',
                           save=True,
                           prefix=prefix,
                           name='elec_price',
                           ylim=(0,80),
                           fliersize=0.1,
                           linewidth=0.5,
                           color='#8f897b'
                           )

plot_elec_price_box()

#%% Box Co2 price

def plot_co2_price_box():
    df = dfs['df_nodal_co2_price'].copy()

    df = df[df.median().sort_values().index]
    df['year'] = dfs['df_chain']['year']
    #df = df[index_order]
    

    optimal_index = dfs['df_chain'].query('year == "scenarios_2030_f"').index
    #optimal_index = optimal_index[:-1]
    #optimal_index = optimal_index.append(dfs['df_chain'].query('s == 1 & c == 1').index)
    df_optimal = df.iloc[optimal_index]
    df_optimal['Scenario'] = np.array(scenarios)

    #df_optimal = df_optimal.iloc[[0,1,3,4]]

    # filter out burnin samples
    df = df[filt_burnin & filt_co2_cap]

    plot_box(df,df_optimal,ylabel='€/T CO$_2$',
                           title='Abetement cost',
                           save=True,
                           prefix=prefix,
                           name='co2_price',
                           ylim=(-2,80),
                           fliersize=0.1,
                           linewidth=0.5,
                           color='#8f897b'
                           )

plot_co2_price_box()

#%% Plot of CO2 reductions 

def plot_co2_reduction_box():
    df = dfs['df_nodal_co2_reduct'].copy()*100

    
    df = df[df.median().sort_values().index]
    df['year'] = dfs['df_chain']['year']
    

    optimal_index = dfs['df_chain'].query('year == "scenarios_2030_f"').index
    #optimal_index = optimal_index[:-1]
    #optimal_index = optimal_index.append(dfs['df_chain'].query('s == 1 & c == 1').index)
    df_optimal = df.iloc[optimal_index]
    df_optimal['Scenario'] = np.array(scenarios)

    #df_optimal = df_optimal.iloc[[0,1,3,4]]


    # filter out burnin samples
    df = df[filt_burnin & filt_co2_cap]

    plot_box(df,df_optimal,ylabel='% CO2 emissions relative to 1990',
                           title='Relative CO2 emissions',
                           save=True,
                           prefix=prefix,
                           name='co2_reduction',
                           ylim=(0,200),
                           whis=(2,98),
                           fliersize=0.1,
                           linewidth=0.5,
                           color='#8f897b'
                           )

plot_co2_reduction_box()

#%%
def plot_co2_box():

    #df = dfs['df_co2']/dfs['df_country_energy']
    df = dfs['df_co2']
    df['year'] = dfs['df_chain'][['year']]

    # assign new sort order 
    #df = df[df.mean().sort_values().index]
    df['year'] = dfs['df_chain']['year']
    df = df[index_order]
    

    optimal_index = dfs['df_chain'].query('year == "scenarios_2030_f"').index
    #optimal_index = optimal_index[:-1]
    optimal_index = optimal_index.append(dfs['df_chain'].query('s == 1 & c == 1').index)
    df_optimal = df.iloc[optimal_index]
    df_optimal['Scenario'] = np.array(['local load','local 1990','Optimum uc','EU ETS','Optimum',])

    # filter out burnin samples
    df = df[filt_burnin & filt_co2_cap]

    plot_box(df,df_optimal,title='Country co2 emmission',save=True,prefix=prefix,name='co2_box')

plot_co2_box()
#%%

def plot_allocated_co2():

    df = dfs['df_theta'].copy()
    df.columns=dfs['mcmc_variables']


    for year in networks:
        co2_budget = networks[year].global_constraints.loc['CO2Limit'].constant
        df.loc[dfs['df_chain'].year == year] = df.loc[dfs['df_chain'].year == year].multiply(co2_budget)

    df = df.iloc[:,:33] #- df_co2.iloc[:,:33]

    df = df[df.mean().sort_values().index]
    df['year'] = dfs['df_chain']['year']
    df = df[filt_burnin & filt_co2_cap]

    #df['year'] = dfs['df_chain']['year']
    plot_box(df,title='Allocated Co2',prefix=prefix,name='allocated_co2',save=True)

plot_allocated_co2()


#%%

def plot_unused_co2():

    df = dfs['df_co2_assigned'].copy()
    df = dfs['df_co2']/df

    df = (df[df.mean().sort_values(ascending=False).index])*100
    df = df.iloc[:,:33]
    df['year'] = dfs['df_chain']['year']
    #df['year'] = dfs['df_chain']['year']

    optimal_index = dfs['df_chain'].query('year == "scenarios_2030_f"').index
    #optimal_index = optimal_index[[-1]]
    #optimal_index = dfs['df_chain'].query('s == 1 & c == 1').index
    #df_optimal = df.iloc[optimal_index]
    #df_optimal['Scenario'] = np.array(['Local Load','Local 1990','Optimum','EU ETS'])
    df_optimal = df.iloc[optimal_index]
    df_optimal['Scenario'] = np.array(scenarios)

    # filter out burnin samples
    df = df[filt_burnin & filt_co2_cap]

    index_order = list(df.mean().sort_values(ascending=False).index)
    index_order.append('year')

    df = df[index_order]

    plot_box(df,df_optimal,#title='Overperformance on national reduction targets',
                ylabel='Fraction of national target used\n[%]',
                prefix=prefix,
                name='unused_co2',
                save=True,
                fliersize=0.1,
                linewidth=0.5,
                color='#8f897b')

plot_unused_co2()

#%% Plot co2 allocated vs el price 

#f,ax = plt.subplots(1,3)

def plot_country_co2_vs_elec_price(countries):
    
    f,ax = plt.subplots(1,5,sharey=True,figsize=(12,3))

    for i,country in enumerate(countries):

        country_idx = np.where(np.array(dfs['mcmc_variables'])==country)[0]
        #plt.scatter((df_theta*base_emis).iloc[:,country_idx],df_country_el_price[country])
        #
        # ax[i].scatter((df_theta*base_emis*0.45).iloc[:,country_idx],dfs['df_co2'][country][:8011],alpha=0.2)

        x = dfs['df_co2_assigned'][country]*1e-6
        x = x[filt_burnin & filt_co2_cap]

        y = (dfs['df_co2'][country].iloc[:-3]/dfs['df_co2_assigned'][country])*100
        y = y[filt_burnin & filt_co2_cap]

        ax[i].scatter(x,y,
                        alpha=0.01,
                        #c='#8f897b'
                        )
        ax[i].set_xticks(np.arange(0, max(x)+1, 20))

        ax[i].set_xlabel('CO$_2$ target\n [M ton CO2]')
        ax[i].set_title(country)
    ax[0].set_ylabel('Fraction of national\ntarget used [%]')
    plt.savefig(f'graphics/co2_vs_co2_{countries}.jpeg',dpi=400,bbox_inches='tight')

plot_country_co2_vs_elec_price(['PL','NL','AT','FI','SE'])


#%% Plot national co2 reduction vs co2 reduction cost

def plot_country_red_vs_co2_price(countries):
    f,ax = plt.subplots(2,int(len(countries)/2),sharey=True,sharex=True,figsize=(12,6))

    for i,country in enumerate(countries):
        df_nodal_co2_reduct = dfs['df_nodal_co2_reduct']*100

        ax[i%2,i%3].scatter(y=dfs['df_nodal_co2_price'][country],x=df_nodal_co2_reduct[country],alpha=0.01 )
        #sns.histplot(x=dfs['df_nodal_co2_price'].values.flatten(),y=df_nodal_co2_reduct.values.flatten(),
        #                bins=60,
        #                binrange=((2,100),(0,1)),
        #                pmax=0.8)
        #
        ax[i%2,i%3].set_xlim(-5,105)
        ax[i%2,i%3].set_title(country)
        ax[i%2,i%3].grid()
        #ax[i].set_xlabel('% CO2 reduction\n relative to 1990 values')
    ax[0,0].set_ylabel('CO2 policy\nprice [€/ton]')
    ax[1,0].set_ylabel('CO2 policy\nprice [€/ton]')
    plt.ylim(0,100)
    
    f.text(0.5, 0.05, '% CO2 emissions relative to 1990 values', ha='center')
    #plt.xlabel('% CO2 reduction relative to 1990 values')

    plt.savefig(f'graphics/red_vs_co2_price_{countries}.jpeg',dpi=400,bbox_inches='tight')

plot_country_red_vs_co2_price(['RO','GB','ES','DE','DK','PL'])


#%% Plot histogram of all co2 prices vs co2 reductions

df_nodal_co2_reduct = dfs['df_nodal_co2_reduct']
sns.histplot(x=dfs['df_nodal_co2_price'].values.flatten(),y=df_nodal_co2_reduct.values.flatten(),
                bins=60,
                binrange=((2,100),(0,1)),
                pmax=0.8)

#%% Plot national co2 reduction vs power price

df_country_el_price = dfs['df_nodal_el_price'].copy()

df_country_el_price.columns = [network.buses.loc[b].country for b in dfs['df_nodal_el_price'].columns]
df_country_el_price = df_country_el_price.iloc[:,df_country_el_price.columns != '']
df_country_el_price = df_country_el_price.groupby(df_country_el_price.columns,axis=1).mean()


countries = ['FR','AT','GB','NL','PL']

f,ax = plt.subplots(1,len(countries),sharey=True,sharex=True,figsize=(12,3))

for i,country in enumerate(countries):
    df_nodal_co2_reduct = dfs['df_nodal_co2_reduct']*100

    ax[i].scatter(y=df_country_el_price[country],x=df_nodal_co2_reduct[country],alpha=0.02 )
    #sns.histplot(x=dfs['df_nodal_co2_price'].values.flatten(),y=df_nodal_co2_reduct.values.flatten(),
    #                bins=60,
    #                binrange=((2,100),(0,1)),
    #                pmax=0.8)
    #
    ax[i].set_xlim(-5,105)
    ax[i].set_title(country)
    #ax[i].set_xlabel('% CO2 reduction\n relative to 1990 values')
ax[0].set_ylabel('Electricity \nprice [€/MWh]')
plt.ylim(0,100)
f.text(0.5, 0.00, '% CO2 reduction relative to 1990 values', ha='center')
#plt.xlabel('% CO2 reduction relative to 1990 values')

#%%

df_nodal_co2_reduct = dfs['df_nodal_co2_reduct']*100
sns.histplot(x=df_country_el_price.values.flatten(),y=df_nodal_co2_reduct.values.flatten(),
                bins=60,
                binrange=((0,100),(0,50)),
                pmax=0.8)



#%%
######## Latex table of brown field technologies

df_renewables = network.generators.query('p_nom_extendable == False').groupby(['country','carrier']).sum().p_nom.unstack(level=1)

df_conventionals =  network.links.query('p_nom_extendable == False & location != ""').groupby(['country','carrier']).sum().p_nom.unstack(level=1)

df = pd.concat([df_renewables,df_conventionals],axis=1)
df.fillna(0,inplace=True)

print(df.to_latex(float_format='{:0.1f}'.format))



#%%################## corrolelogram tech energy #######################
#######################################################################

def plot_corrolelogram_tech_energy(prefix='',save=False,plot_optimum=False,
    technologies = {'wind':['offwind','offwind-ac','offwind-dc','onwind'],
                    'lignite + coal' : ['lignite','coal'],
                    'gas': ['OCGT','CCGT'],
                    'solar':['solar','solar rooftop'],
                    'nuclear':['nuclear'],
                    #'oil':['oil'],
                    #'H2':['H2 Fuel Cell','H2 Electrolysis','H2_store'],
                    #'H2':['H2 Fuel Cell',],
                    #'battery':['battery charger','battery discharger','battery_store']
                    #'battery':['battery discharger']
                    },
                    title='Technology energy production'):

    df = dfs['df_chain'][['year']]
    for key in technologies: 
        #df[key] = df_tech_e_sum[technologies[key]].sum(axis=1)
        df[key] = dfs['df_energy'].loc[:,(slice(None),slice(None),technologies[key])].sum(axis=1)

    if plot_optimum:
        optimum_index = np.where(dfs['df_chain'].s == 1)[0][0]
        df_optimum = df.iloc[[17]]


    # filter out burnin samples
    df = df[dfs['df_chain'].s>burnin_samples]

    sns_plot = sns.pairplot(df, kind="hist", diag_kind='hist',
                            #hue='year',
                            plot_kws=dict(bins=30),
                            diag_kws=dict(bins=40,),
                            palette='Set1'
                            )

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



#%%

technologies = {'wind':['offwind','offwind-ac','offwind-dc','onwind'],
                'lignite + coal' : ['lignite','coal'],
                'gas': ['OCGT','CCGT'],
                'solar':['solar','solar rooftop'],
                'nuclear':['nuclear'],
                }
df = dfs['df_chain'][['year']]
for key in technologies: 
    #df[key] = df_tech_e_sum[technologies[key]].sum(axis=1)
    df[key] = dfs['df_energy'].loc[:,(slice(None),slice(None),technologies[key])].sum(axis=1)

df_long = pd.melt(df,value_vars=technologies.keys(),id_vars='year',var_name='Tech')

ax = sns.violinplot(x="Tech", y="value", data=df_long,cut=0,scale='width')
#plt.ylim(0,0.8e6)

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

    df = dfs['df_chain'][['year']]
    for key in technologies:
        df[key] = dfs['df_tech_sum'][technologies[key]].sum(axis=1)


    if plot_optimum:
        optimum_index = np.where(dfs['df_chain'].s == 1)[0][0]
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

    df = dfs['df_chain'][['year']]
    for key in metrics: 
        df[key] = dfs['df_secondary'][metrics[key]].sum(axis=1)


    if plot_optimum:
        #optimum_index = np.where(dfs['df_chain'].s == 1)[0][0]
        optimal_index = dfs['df_chain'].query('year == "scenarios"').index
        optimal_index = optimal_index.append(dfs['df_chain'].query('s == 1 & c == 1').index)
        #df_optimal = df.iloc[optimal_index]
        #dfs['df_secondary'].iloc[[17],:]
        df_optimum = dfs['df_chain'].iloc[optimal_index][['year']]
        for key in metrics:
            df_optimum[key] = dfs['df_secondary'].iloc[optimal_index][metrics[key]].sum(axis=1)


    filt_low_co2_red = dfs['df_secondary'].co2_reduction<=70
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
    
dfs['df_secondary']['energy dependance'] = dfs['df_energy_dependance'] 
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




#%% Correlation of CO2 allocations

def plot_co2_correlation_matrix():

    f, axes = plt.subplots( figsize=(15, 11), sharex=True, sharey=False)
    df_co2_assigned_noEU = dfs['df_co2_assigned'].iloc[:,:-1]

    corr = dfs['df_co2'][dfs['df_chain'].s>burnin_samples].corr()
    #corr = df_co2_assigned_noEU[dfs['df_chain'].s>burnin_samples].corr()
    #corr = (df_co2/df_co2_assigned_noEU ).corr()

    sns.heatmap(corr, 
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                square=True,
                vmin=-1,
                cmap='RdYlGn',
                #ax = ax
                )
    plt.title('CO2 emissions')
    plt.savefig(f'graphics/co2_emis_correlations_{prefix}.jpeg',dpi=400)
plot_co2_correlation_matrix()

#%%

def heatmap(x, y, size):
    fig, ax = plt.subplots(figsize=(12,12))
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 

    n_colors = 256 # Use 256 colors for the diverging color palette
    palette = sns.diverging_palette(20, 220, n=n_colors) # Create the palette
    color_min, color_max = [-1, 1] # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
        ind = int(val_position * (n_colors - 1)) # target index in the color palette
        return palette[ind]

    #color = [value_to_color for ]
    
    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num), # Use mapping for x
        y=y.map(y_to_num), # Use mapping for y
        s=size.abs() * size_scale, # Vector of square sizes, proportional to size parameter
        c=size.apply(value_to_color), # Vector of square color values, mapped to color palette
        marker='o' # Use square as scatterplot marker
    )
    
    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=90, horizontalalignment='center')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)
    
corr = dfs['df_co2'][dfs['df_chain'].s>burnin_samples].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
corr.loc[corr.query('x == y').index,'value'] = 0
heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value']
)

plt.savefig(f'graphics/co2_emis_correlations_{prefix}.jpeg',dpi=400)

#%% CO2 emission correlellogram 

#Germany, Czech Republic, Slovakia, Poland, Hungary, Slovenia, Croatia, Liechtenstein, and Switzerland)

def plot_country_co2_correlation(countries):

    sns.pairplot(dfs['df_co2'].loc[dfs['df_chain'].s>burnin_samples,countries],kind='hist')
    #plt.title('CO2 emissions')
    plt.savefig(f'graphics/co2_correllelogram_{countries}_{prefix}.jpeg',dpi=400)

plot_country_co2_correlation(['PL','LV','DE'])


#%% Plot map with CO2 correlations

#df = dfs['df_nodal_el_price'].copy()
#
#df.columns = [network.buses.loc[b].country for b in dfs['df_nodal_el_price'].columns]
#df = df.iloc[:,df.columns != '']
#df = df.groupby(df.columns,axis=1).mean()
#
#corr = df.corr()

#corr = dfs['df_nodal_co2_price'][dfs['df_chain'].s>burnin_samples].corr()

corr = dfs['df_co2'][dfs['df_chain'].s>burnin_samples].corr()

countries = np.array(corr.index)

bus1 = []
bus2 = []
names = []
weight = []
for c1 in countries: 
    for c2 in countries: 
        if c1 != c2:
            bus1.append(network.buses.query(f'country == "{c1}"').iloc[0].name)
            bus2.append(network.buses.query(f'country == "{c2}"').iloc[0].name)
            names.append(c1+'-'+c2)
            weight.append(corr.loc[c1,c2])

n_test = network.copy()
#n_test.carriers = network.carriers
#
buses = network.buses.query('carrier == "AC"')
buses.index = buses.country
buses.drop_duplicates(inplace=True)
#
#n_test.buses = buses

n_test.mremove('Bus',list(network.buses.query('carrier != "AC"').index))

n_test.mremove('Link',list(network.links.index))

n_test.madd('Link',
            bus0 = bus1,
            bus1 = bus2,
            names= names,
            p_nom = weight)

cmap = matplotlib.cm.get_cmap('RdYlGn')

n_colors = 256 # Use 256 colors for the diverging color palette
palette = sns.diverging_palette(20, 220, n=n_colors) # Create the palette
color_min, color_max = [-1, 1] # Range of values that will be mapped to the palette, i.e. min and max possible correlation

def value_to_color(val):
    val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
    ind = int(val_position * (n_colors - 1)) # target index in the color palette
    return palette[ind]

colors = [value_to_color(l[1].p_nom) for l in n_test.links.iterrows()]
size_scaling = lambda x: (abs(x)**1.5)*10
sizes = (abs(n_test.links.p_nom)**1.5)*10
sizes[sizes<size_scaling(0.2)]=0

fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
fig.set_size_inches(9, 9)
fig.patch.set_visible(False)
ax.axis('off')
n_test.plot(
        #bus_sizes=bus_cap/bus_size_factor,
        bus_colors='black',
        #line_colors=ac_color,
        #link_colors=[cmap(l[1].p_nom) for l in n_test.links.iterrows()],
        link_colors=colors,
        link_widths=sizes,
        line_widths=0,
        #line_colors='green',
        #link_widths=link_width/linewidth_factor,
        #ax=ax[int(np.floor(i/2)),i%2],  
        boundaries=(-10, 30, 34, 70),
        color_geomap={'ocean': 'white', 'land': (203/255, 203/255, 203/255)}
        )

#plt.title('CO2 emission correlations')
data = np.linspace(-1,1, 100).reshape(10, 10)
cax = fig.add_axes([0.87, 0.15, 0.025, 0.7])
im = ax.imshow(data, cmap=matplotlib.colors.ListedColormap(palette),visible=False)
fig.colorbar(im, cax=cax, orientation='vertical')


plt.savefig(f'graphics/co2_correlation_map_{prefix}.jpeg',transparent=True,dpi=400)

#%% Plot CO2 price vs reduction correlation 


def plot_country_co2_price_vs_emis(country):
#df = pd.DataFrame( data={'CO2 price':df_nodal_co2_price[country],'Assigned quotas':df_theta[country]*base_emis})
    df = pd.DataFrame( data={'CO2 policy price':dfs['df_nodal_co2_price'][country],'CO2 emissions':dfs['df_nodal_co2_reduct'][country]*100})
    #df = pd.DataFrame( data={'CO2 price':dfs['df_country_cost'][country],'Assigned quotas':dfs['df_co2'][country]})

    #sns.lmplot(data = df[filt_burnin & filt_co2_cap], x='CO2 emissions' , y='CO2 policy price',line_kws={'color':'r'},scatter_kws={'alpha':0.1})
    sns.scatterplot(data = df[filt_burnin & filt_co2_cap], x='CO2 emissions' , y='CO2 policy price',alpha=0.1)

    plt.title(country)
    plt.ylim(0,100)
    plt.xlabel('% CO2 emissions relative to 1990')
    #plt.xlim(0,0.5e7)

country = 'RO'
plot_country_co2_price_vs_emis(country)

#%% Plot of brownfield capacities


def plot_brownfield_capacities():
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
    tech_colors = get_tech_colors()


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

plot_brownfield_capacities()


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

#%% Distribution of installed capacity on country level 

def plot_generator_variance():

    plt.subplots(figsize=(14,5))
    df_gen_grouped = dfs['df_gen_p'].groupby([network.generators.carrier,network.generators.country],axis=1).sum()

    df_gen_grouped.drop(df_gen_grouped.loc[:,(slice(None),'')].columns,axis=1,inplace=True)

    #df_gen_grouped.loc[:,('solar','ES')].hist()

    ax = sns.boxplot(data=df_gen_grouped.loc[:,df_gen_grouped.var()>1e2]*1e-3,
                whis=(0,100),
                )

    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
    plt.ylabel('Installed capacity [GW]')
    plt.xlabel('Generator')
    plt.title('Generator capacity with variance > 100 MW')

    plt.savefig(f'graphics/generator_variance_{prefix}.jpeg',dpi=fig.dpi,bbox_inches='tight')

plot_generator_variance()



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
    sns.histplot(x=df_secondary['system_cost'],y=dfs['df_co2'][country])

    plt.savefig(f'graphics/country_co2_co2_{country}_{prefix}.jpeg')

plot_country_cost_co2(country='DE')

#%%##################### co2 emis pr gdp geo plot #############
###############################################################

def plot_co2_gpd_geo(prefix='',save=False,year=2030):
    model_countries = network.buses.country.unique()[:33]
    alpha3 = [iso_countries.get(c).alpha3 for c in model_countries]
    df_gdp_i = df_gdp.set_index('Country Code')
    model_countries_gdp = pd.DataFrame(df_gdp_i.loc[alpha3]['2018'])
    model_countries_gdp.index = model_countries

    co2_pr_gdp = dfs['df_co2'].divide(model_countries_gdp['2018'],axis=1)
    co2_pr_gdp['year'] = df_chain['year']


    df = co2_pr_gdp.query(f'year == {year}').mean()[:33]*1e6

    #df = df_country_el_price.iloc[7750]
    #df = df_nodal_co2_price.iloc[36]
    #df = dfs['df_co2_pr_energy'].iloc[27]
    df = (dfs['df_co2_sweep']/(df_country_load*network.snapshot_weightings[0])*1000).iloc[0,:-1]

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

    co2_pr_pop = dfs['df_co2'].divide(model_countries_pop['2018'],axis=1)
    co2_pr_pop['year'] = df_chain['year']


    df = co2_pr_pop.query(f'year == {year}').mean()[:33]*1e6

    fig = plot_geo(df,title)
    if save:
        fig.write_image(f'graphics/co2_pop_geo_{year}_{prefix}.jpeg')

plot_co2_pop_geo(year=2050,save=True,prefix=prefix,title='Mean co2 emis pr population 2050')


#%% Plot of chain development over time 
def plot_chain_development(prefix='',save=False):
    accept_percent = sum(dfs['df_chain'].a)/dfs['df_theta'].shape[0]*100
    print(f'Acceptance {accept_percent:.1f}%')

    df = dfs['df_theta'].copy()
    df['index'] = df.index
    df['s'] = dfs['df_chain']['s']
    df['c'] = dfs['df_chain']['c']

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
    x = dfs['df_chain'].groupby(dfs['df_chain'].s).mean().a
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
    series = dfs['df_theta'].iloc[dfs['df_chain'].iloc[0:-3].query(f'c == {chain}').index][str(theta)]
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
