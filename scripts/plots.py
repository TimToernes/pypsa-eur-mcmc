"""
Script for generating the plots used in the manuscript 
"30.000 ways to reach 55% decarbonization of the European electricity supply"
"""

#%%
import pypsa
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Circle, Ellipse
from matplotlib.legend_handler import HandlerPatch, HandlerBase
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import cm
import matplotlib as mpl
import seaborn as sns
from sklearn.ensemble import ExtraTreesRegressor
import cartopy.crs as ccrs
import pytz
import iso3166
from _mcmc_helpers import read_csv
from _helpers import make_override_component_attrs, get_tech_colors
from data_process import data_postprocess


if os.path.split(os.getcwd())[1] == 'scripts':
    os.chdir('..')

#%%#################### import datasets ####################################

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


prefix = '2030_elec_f'
burnin_samples = 100
base_emis = 1481895952.8
datasets = ['mcmc_2030_f', 'sweep_2030_f', 'scenarios_2030_f','scenarios_sweep_2030_f']
scenarios = ['Grandfathering', 'Sovereignty', 'Efficiency', 'Egalitarianism', 'Ability to pay']

dfs,networks = import_datasets(datasets)
network = list(networks.values())[0]

#%%##### Data postprocessing #################
##############################################

data_postprocess(dfs,networks,base_emis,co2_red=0.45,scenario_names="scenarios_2030_f")

# Filters 
filt_co2_cap = dfs['df_co2'].sum(axis=1)<=base_emis*0.448
filt_burnin = dfs['df_chain'].s>burnin_samples

# Replace the first sample in the chains with the optimal solution
i_sweep = dfs['df_chain'].query('year == "sweep_2030_f"').index
i_optimum = dfs['df_secondary'].iloc[i_sweep].query('co2_reduction <= 55 & co2_reduction >= 54.999').index 
i_s1 = dfs['df_chain'].query('s==1').index
for df in dfs:
    if df not in ['mcmc_variables','df_country_pop','df_gdp','df_pop','df_country_gdp','df_country_load','df_chain']:
        dfs[df].iloc[i_s1] = dfs[df].iloc[i_optimum]


#%%############# plots #############################
# ############################################


# ####### Corrolelogram cost vs co2 #########
"""
Plot of samples CO2 reduction against cost increase
"""

def plot_cost_vs_co2(prefix='',save=False,
                    plot_sweep=False,
                    plot_optimum=False,
                    plot_scenarios=False,
                    sweep_name='sweep_2030_f',
                    co2_emis_level = 55,
                    scenarios_style = 'original'):
    
    # Create dataframe with relevant data
    df = dfs['df_secondary'][['cost_increase']]
    df['co2 emission'] =dfs['df_store_P']['co2 atmosphere']
    df['co2 reduction'] = 1-(df['co2 emission']/base_emis )
    df['co2 reduction'] = df['co2 reduction']*100
    df['year'] = dfs['df_chain']['year']

    co2_label = 'CO$_2$ reduction [%]'
    cost_label = 'Cost increase [%]'

    df.rename(columns={'cost_increase':cost_label,'co2 reduction':co2_label},inplace=True)

    # Create dataframe with optimal solution
    if plot_optimum:
        index_optimum = dfs['df_chain'].query('c==1 & s==1').index
        #index_optimum = dfs['df_chain'].query('year == "scenarios_2030_f"').index
        df_optimum = df.iloc[index_optimum]

    if plot_sweep:
        index_sweep = dfs['df_chain'].query(f'year == "{sweep_name}"').index
        df_sweep = df.iloc[index_sweep]

    if plot_scenarios:

        
        scheme_encoding = {'Grandfathering':1,'Sovereignty':2,'Efficiency 55%':3,'Egalitarianism':4,'Ability to pay':5,'Efficiency 70%':6}
        ivd = {v: k for k, v in scheme_encoding.items()}

        if scenarios_style == 'original':

            index_scenarios = dfs['df_chain'].query(f'year == "scenarios_2030_f"').index
            df_scenarios = df.iloc[index_scenarios]
            df_scenarios['Scenario'] = [ivd[c] for c in  dfs['df_chain'].iloc[index_scenarios].c]

        elif scenarios_style == 'sweep':

            index_scenarios = dfs['df_chain'].query(f'year == "scenarios_sweep_2030_f"').index
            df_scenarios = df.iloc[index_scenarios]
            df_scenarios['Scenario'] = [ivd[c] for c in  dfs['df_chain'].iloc[index_scenarios].c]

        
        elif scenarios_style == 'all55p':
            
            index_scenarios = dfs['df_chain'].query(f'year == "scenarios_sweep3_2030_f" | year == "scenarios_2030_f"').index
            index_new_scenarios = dfs['df_secondary'].iloc[index_scenarios].query(f'54.99 < co2_reduction < 55.101').index
            df_scenarios = df.iloc[index_new_scenarios]

            df_scenarios['Scenario'] = [ivd[c] for c in  dfs['df_chain'].iloc[index_new_scenarios].c]
            df_scenarios.drop_duplicates('Scenario',inplace=True)

    def scenarios_plot():
        scenario_names = scenarios
        scenario_names = df_scenarios['Scenario'].unique()
        x = df_scenarios[co2_label]
        y = df_scenarios[cost_label]
        if scenarios_style == 'original':
            sns_plot.ax_joint.scatter(x,
                        y,
                        c = '#f58905',
                        s = 50,
                            )
        for i,txt in enumerate(scenario_names):
            if scenarios_style == 'sweep':
                x = df_scenarios.query(f"Scenario=='{txt}'")[co2_label]
                y = df_scenarios.query(f"Scenario=='{txt}'")[cost_label]
                sns_plot.ax_joint.plot(x,y,label=txt)
            if txt == 'Egalitarianism' :
                offset = 0.5
            elif txt == 'Ability to pay' :
                offset = -0.9
            else :
                offset = 0 
            if scenarios_style == 'original':
                sns_plot.ax_joint.annotate(txt, (x.iloc[i], y.iloc[i]+offset),fontsize=12)
            sns_plot.ax_joint.legend()
    df = df[ filt_burnin ]
    sns_plot = sns.jointplot(data=df,
                   x=co2_label,
                   y=cost_label,
                   kind='hist',
                   joint_kws={'bins':50,'thresh':0},
                   marginal_ticks=True,
                    )

    sns_plot.ax_joint.axvline(co2_emis_level,c='r')
    sns_plot.ax_joint.axhline(18,c='r')

    if plot_sweep:
        sns_plot.ax_joint.plot(df_sweep[co2_label],
                                df_sweep[cost_label],
                                markersize=10)

    # Draw optimal solution on plot 
    if plot_optimum:
        sns_plot.ax_joint.plot(df_optimum[co2_label],
                                df_optimum[cost_label],
                                        marker='X',
                                        mfc='r',
                                        markersize=20)

    # Draw optimal solution on plot 
    if plot_scenarios:
        scenarios_plot()


    sns_plot.ax_joint.set_xlim((54,76))
    sns_plot.ax_joint.set_ylim((-1,19))

    if scenarios_style == 'sweep':
        prefix = 'sweep' + prefix

    if save:
        sns_plot.savefig(f'graphics/cost_vs_co2_{prefix}.pdf')


plot_cost_vs_co2(save=True,prefix=prefix,
                        plot_sweep=True,
                        plot_optimum=False,
                        plot_scenarios=True,
                        scenarios_style='original')


plot_cost_vs_co2(save=True,prefix=prefix,
                        plot_sweep=True,
                        plot_optimum=False,
                        plot_scenarios=True,
                        scenarios_style='sweep')

#%%############ Plot Boxplots ############################
"""
Function definitions for the boxplots style used
"""

def make_legend(list_color  = ["c", "gold", "crimson"],
                list_mak    = ["d","s","o"],
                list_lab    = ['Marker 1','Marker 2','Marker 3']):

    ax = plt.gca()

    class MarkerHandler(HandlerBase):
        def create_artists(self, legend, tup,xdescent, ydescent,
                            width, height, fontsize,trans):
            return [plt.Line2D([width/2], [height/2.],ls="",
                        marker=tup[1],color=tup[0], transform=trans)]

    ax.legend(list(zip(list_color,list_mak)), list_lab,#loc='upper left',
            handler_map={tuple:MarkerHandler()}) 



def plot_box(df_wide,df_wide_optimal=None,prefix='',save=False,title='',name='co2_box',ylabel='CO2 emission',loc=None,ylim=None,**kwargs):
    model_countries = network.buses.country.unique()[:33]
    df = pd.melt(df_wide,value_vars=model_countries,id_vars='year',var_name='Country')

    f,ax = plt.subplots(figsize=(11,4))
    sns_plot = sns.boxplot(x='Country', y="value", 
                        data=df, 
                        #palette="muted",
                        order=df_wide.columns[:-1],
                        ax=ax,
                        **kwargs)

    if df_wide_optimal is not None:
        
        list_color =  {'Grandfathering':'#88E0EF',
                        'Sovereignty':'#161E54',
                        'Efficiency 55%':'#FF5151',
                        'Egalitarianism':'#B000B9',
                        'Ability to pay':'#FF9B6A',
                        'Efficiency 70%':'#4287f5',}                        
        
        df_optimal = pd.melt(df_wide_optimal,value_vars=model_countries,id_vars=['year','Scenario'],var_name='Country')

        sns.stripplot(x='Country',y='value',hue='Scenario',
                        data=df_optimal,
                        order=df_wide.columns[:-1],
                        jitter=0.15,
                        dodge=True,
                        linewidth=0,
                        alpha=0.9,
                        palette=list_color,
                        size=5,
                        ax=ax,)
        
    plt.ylabel(ylabel)
    plt.gca().set_title(title, y=1.0, pad=-14)
    
    if loc != None:
        plt.legend(loc=loc)

    if ylim != None:
        plt.ylim(ylim)

    if save:
        plt.savefig(f'graphics/{name}_{prefix}.jpeg',transparent=False,dpi=400)

#%% Set index order for box plots

index_order_co2mwh = (dfs['df_co2']/dfs['df_country_energy']).mean().sort_values().index

if not 'year' in index_order_co2mwh:
    index_order_co2mwh = index_order_co2mwh.append(pd.Index(['year']))

# Optimal index original 
optimal_index = dfs['df_chain'].query('year == "scenarios_2030_f"').index

scheme_encoding = {'Grandfathering':1,'Sovereignty':2,'Efficiency 55%':3,'Egalitarianism':4,'Ability to pay':5,'Efficiency 70%':6}
ivd = {v: k for k, v in scheme_encoding.items()}

scenarios = list(dfs['df_chain'].iloc[optimal_index].c.map(ivd).values)

#%%
"""
Boxplot of relative CO2 intesity for all model countries 
"""

def plot_co2_pr_mwh_box():

    n_hours_per_snapshot = 3
    df = dfs['df_co2']/(dfs['df_country_energy']*n_hours_per_snapshot)
    df = df*1e3
    df['year'] = dfs['df_chain']['year']

    # Rearange collumns to match index order 
    df = df[index_order_co2mwh]
    df['year'] = dfs['df_chain']['year']

    df_optimal = df.iloc[optimal_index]
    df_optimal['Scenario'] = np.array(scenarios)

    # filter out burnin samples
    df = df[filt_burnin & filt_co2_cap]

    plot_box(df,df_optimal,ylabel='Relative CO$_2$ intensity\n[gCO$_2$/kWh]',
                           title='',
                           save=True,
                           prefix=prefix,
                           name='co2_mwh_box',
                           fliersize=0.1,
                           linewidth=0.5,
                           color='#bdbdbd'
                           )
    plt.gca()


plot_co2_pr_mwh_box()


#%% Box elec price
"""
Boxplot of mean national electricity prices for all model nations
"""
def plot_elec_price_box():
    df = dfs['df_nodal_el_price'].copy()
    
    df.columns = [network.buses.loc[b].country for b in dfs['df_nodal_el_price'].columns]
    df = df.iloc[:,df.columns != '']
    df = df.groupby(df.columns,axis=1).mean()

    df = df[df.median().sort_values().index]
    # Rearange collumns to match index order 
    df['year'] = dfs['df_chain']['year']

    df_optimal = df.iloc[optimal_index]
    df_optimal['Scenario'] = np.array(scenarios)

    df = df[filt_burnin & filt_co2_cap]
    df.reindex()

    plot_box(df,df_optimal,ylabel='Electricity price\n[€/MWh]',
                           title='',
                           save=True,
                           #loc=('upper left'),
                           prefix=prefix,
                           name='elec_price',
                           ylim=(0,80),
                           fliersize=0.1,
                           linewidth=0.5,
                           color='#bdbdbd'
                           )
    

plot_elec_price_box()

#%% Box Co2 price
"""
Boxplot of CO2 abatement cost for all model nations
"""

def plot_co2_price_box():
    df = dfs['df_nodal_co2_price'].copy()

    df = df[df.mean().sort_values().index]
    df['year'] = dfs['df_chain']['year']

    df_optimal = df.iloc[optimal_index]
    df_optimal['Scenario'] = np.array(scenarios)

    # filter out burnin samples
    df = df[filt_burnin & filt_co2_cap]

    plot_box(df,df_optimal,ylabel='Abetement cost\n[€/tCO$_2$]',
                           title='',
                           save=True,
                           prefix=prefix,
                           name='co2_price',
                           ylim=(-2,80),
                           fliersize=0.01,
                           linewidth=0.5,
                           color='#bdbdbd'
                           )

plot_co2_price_box()

#%% Graphical abstract 
"""
Map plot showing abatement costs across all model countries 
Used in graphical abstract
"""
def plot_graphical_abstract_map():

    cmap = cm.get_cmap('cividis_r')
    fig, ax = plt.subplots(1,1,figsize=(10,10)) # Initialise figure
    m_plot = Basemap(width=11500000/2.9,height=9000000/2.2,projection='laea',
                    resolution='i',lat_0=54.5,lon_0=9.5,ax=ax)

    norm = plt.Normalize(vmin=-10, vmax=51)

    countries = network.buses.country.unique()
    countries = countries[:-1]
    countries =  countries[countries !='ME']

    for c in countries:
        alph3 = iso3166.countries[c].alpha3
        m_plot.readshapefile('shapefiles/gadm36_' + alph3 + '_0',c,drawbounds=True,linewidth = 0,color='k')
        patches = []
        scenario_id = dfs['df_chain'].query('year=="scenarios_2030_f" & c==2').index[0]
        value = dfs['df_nodal_co2_price'][c].iloc[scenario_id]
        for info, shape in zip(eval('m_plot.' + c + '_info'), eval('m_plot.' + c)):
            patches.append(Polygon(np.array(shape), True))
        patch1=ax.add_collection(PatchCollection(patches, facecolor= cmap(norm(value))))

    plt.axis('off')

    # Add colorbar 
    cb_ax = fig.add_axes([0.15,0.1,0.8,0.02])
    cb1 = mpl.colorbar.ColorbarBase(cb_ax,orientation='horizontal', cmap=cmap,norm=norm,boundaries=np.arange(0,51),ticks=np.arange(0,51,10)) #,t
    cb1.ax.set_xticklabels(np.arange(0,51,10),fontsize=14)
    cb1.set_label(r'Mean CO$_2$ abatement costs'+' [€/ton]',zorder=10,fontsize=20)

    fig.savefig(f'graphics/graph_abstract_map.jpeg',dpi=300,bbox_inches='tight')  

plot_graphical_abstract_map()


#%%######## Latex table of brown field technologies
"""
Generate LaTex table of browfield technology data
"""

df_renewables = network.generators.query('p_nom_extendable == False').groupby(['country','carrier']).sum().p_nom.unstack(level=1)

conventionals = network.links.query('p_nom_extendable == False & location != ""')
conventionals.loc[:,'p_nom'] = conventionals['p_nom'] * conventionals['efficiency']

df_conventionals =  conventionals.groupby(['country','carrier']).sum().p_nom.unstack(level=1)

df = pd.concat([df_renewables,df_conventionals],axis=1)
df.fillna(0,inplace=True)

print(df.to_latex(float_format='{:0.1f}'.format))

#%% Plot of brownfield capacities
"""
Map plot of brownfield capacities
"""
def plot_brownfield_capacities():
    bus_size_factor = 80000
    linewidth_factor = 2000

    # Get pie chart sizes for technology capacities 
    tech_types = list(network.generators.query('p_nom_extendable == False').carrier.unique()) + list(network.links.query('p_nom_extendable == False').carrier.unique()) + list(network.storage_units.query('p_nom_extendable == False').carrier.unique())
    tech_types.remove('DC')

    bus_cap = pd.Series()
    bus_cap.index = pd.MultiIndex.from_arrays([[],[]],names=['bus','tech'])
    for tech in tech_types:
        s = network.generators.query(f'carrier == "{tech}" & p_nom_extendable == False').p_nom_opt.groupby(network.generators.bus).sum()

        if len(s)<=1:
            links_tech = network.links.query(f'carrier == "{tech}" & p_nom_extendable == False')
            links_tech = links_tech['p_nom'] * links_tech['efficiency']
            s = links_tech.groupby(network.links.bus1).sum()
            stores_tech = network.storage_units.query(f'carrier == "{tech}" & p_nom_extendable == False')
            stores_tech = stores_tech['p_nom'] * stores_tech['efficiency_dispatch']
            s = stores_tech.groupby(network.storage_units.bus).sum()

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

    tech_colors = get_tech_colors()
    network.plot(
            bus_sizes=bus_cap/bus_size_factor,
            bus_colors=tech_colors,
            link_colors='blue',
            line_widths=network.lines.s_nom / linewidth_factor,
            line_colors='#2ca02c',
            link_widths=link_width/linewidth_factor, 
            boundaries=(-10, 30, 34, 70),
            color_geomap={'ocean': 'white', 'land': (203/255, 203/255, 203/255)})

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

    plt.savefig(f'graphics/brownfield_tech{prefix}.pdf',dpi=fig.dpi,bbox_extra_artists=(l2,l1),bbox_inches='tight')


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


plot_brownfield_capacities()

#%% Geographical Potentials 
"""
Plot of geographical renewable potentials
"""
def plot_geographical_potentials():
    bus_size_factor = 80000
    linewidth_factor = 2000
    # Get pie chart sizes for technology capacities 
    tech_types =  list(network.generators.query('p_nom_max < 1e9').carrier.unique())
    tech_colors = get_tech_colors()

    bus_cap = pd.Series()
    bus_cap.index = pd.MultiIndex.from_arrays([[],[]],names=['bus','tech'])
    for tech in tech_types:
        s = (network.generators_t.p_max_pu[network.generators.query(f'carrier == "{tech}" & p_nom_extendable == True').index].mean() * network.generators.query(f'carrier == "{tech}" & p_nom_extendable == True').p_nom_max).groupby(network.generators.bus).sum()
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

    network.plot(
            bus_sizes=bus_cap/bus_size_factor,
            bus_colors=tech_colors,
            #line_colors=ac_color,
            link_colors='blue',
            line_widths=network.lines.s_nom*0 / linewidth_factor,
            line_colors='#2ca02c',
            link_widths=link_width*0/linewidth_factor,
            #ax=ax[int(np.floor(i/2)),i%2],  
            boundaries=(-10, 30, 34, 70),
            color_geomap={'ocean': 'white', 'land': (203/255, 203/255, 203/255)})

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

    plt.savefig(f'graphics/geographic_potentials_{prefix}.pdf',dpi=fig.dpi,bbox_extra_artists=(l2,l1),bbox_inches='tight')

plot_geographical_potentials()

#%%####### Heatmap function 
"""
Function for generating heatmap plots of correlations/impact factors
"""

def heatmap(x, y, size):
    fig, ax = plt.subplots(figsize=(7,7))
    
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
    
    size_scale = 200
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
    

#%% Plot map with correlations
"""
Map plot with arrows for correlations
"""

def correlation_map(corr):
    countries = np.array(corr.index)

    n_test = network.copy()    
    n_test.mremove('Bus',['GB4 0','DK3 0','ES2 0'])

    bus1 = []
    bus2 = []
    names = []
    weight = []
    for c1 in countries: 
        for c2 in countries: 
            if c1 != c2:
                bus1.append(n_test.buses.query(f'country == "{c1}"').iloc[-1].name)
                bus2.append(n_test.buses.query(f'country == "{c2}"').iloc[-1].name)
                names.append(c1+'-'+c2)
                weight.append(corr.loc[c1,c2])

    
    #n_test.carriers = network.carriers
    buses = network.buses.query('carrier == "AC"')
    buses.index = buses.country
    buses.drop_duplicates(inplace=True)

    n_test.mremove('Bus',list(network.buses.query('carrier != "AC"').index))
    n_test.mremove('Link',list(network.links.index))
    n_test.madd('Link',
                bus0 = bus1,
                bus1 = bus2,
                names= names,
                p_nom = weight)

    cmap = matplotlib.cm.get_cmap('RdYlGn')

    n_colors = 256 # Use 256 colors for the diverging color palette
    #palette = sns.diverging_palette(20, 220, n=n_colors) # Create the palette
    #palette=sns.color_palette("ch:start=.2,rot=-.3", n_colors=n_colors)
    palette = sns.diverging_palette(20, 220, n=n_colors*2)[n_colors:]

    color_min, color_max = [0, 1] # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
        ind = int(val_position * (n_colors - 1)) # target index in the color palette
        return palette[ind]

    colors = [value_to_color(l[1].p_nom) for l in n_test.links.iterrows()]
    size_scaling = lambda x: (abs(x)**1.5)*10
    sizes = (abs(n_test.links.p_nom)**1.5)*10
    sizes[sizes<size_scaling(0.2)]=0

    flow= pd.Series(0,index=n_test.branches().index)
    flow.loc['Link',sizes.index] = -sizes.values*10

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
            flow=flow,
            #line_colors='green',
            #link_widths=link_width/linewidth_factor,
            #ax=ax[int(np.floor(i/2)),i%2],  
            boundaries=(-10, 30, 34, 70),
            color_geomap={'ocean': 'white', 'land': (220/255, 220/255, 220/255)}
            #color_geomap={'ocean': '#aed8e5ff', 'land': (203/255, 203/255, 203/255)}
            )

    #plt.title('CO2 emission correlations')
    data = np.linspace(0,1, 100).reshape(10, 10)
    cax = fig.add_axes([0.87, 0.15, 0.025, 0.7])
    im = ax.imshow(data, cmap=matplotlib.colors.ListedColormap(palette),visible=False)
    fig.colorbar(im, cax=cax, orientation='vertical',)


#%%############# plots of impact factors #####################################
"""
Plot the impact factors of CO2 targets on electricity price
"""

def calc_impact_factors():
    """
    Calculate impact factos of CO2 target on electricity price 
    using the Extra Trees regressor 
    """
    countries = dfs['df_co2'].columns

    coefficients = np.empty([0,33])
    scores = []

    for country in countries:

        X = dfs['df_co2_assigned'].values[:,:-1]
        X = X[filt_burnin & filt_co2_cap]

        #y = dfs['df_nodal_co2_price'][country].values
        idx = network.buses.query('carrier=="AC"').index
        y = dfs['df_nodal_el_price'][idx]
        y = y.filter(like=country,axis=1).values
        if y.shape[1]==2:
            y = y.mean(axis=1)
        y= y[filt_burnin & filt_co2_cap]

        # ExtraTreesRegressor
        model = ExtraTreesRegressor(n_estimators=10,bootstrap=True, random_state=0).fit(X,y)
        scores.append(model.score(X,y))
        coef = model.feature_importances_
        coefficients = np.append(coefficients,[coef],axis=0)
    return coefficients


def plot_impact_factors(coefficients):
    countries = dfs['df_co2'].columns
    
    corr = pd.DataFrame(data=coefficients,columns=countries,index=countries)
    corr_long = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
    corr_long.columns = ['x', 'y', 'value']
    #corr.loc[corr.query('x == y').index,'value'] = 0
    heatmap(
        x=corr_long['y'],
        y=corr_long['x'],
        size=corr_long['value']
    )
    plt.xlabel(r'CO$_2$ emissions allowed')
    #plt.ylabel(r'Nodal electricity price')
    plt.ylabel(r'Mean nodal electricity price')
    plt.savefig(f'graphics/impact_co2_vs_el_price_{prefix}.pdf')

coefficients = calc_impact_factors()
plot_impact_factors(coefficients)

#%%
countries = dfs['df_co2'].columns
corr = pd.DataFrame(data=coefficients,columns=countries,index=countries)
correlation_map(corr)
plt.savefig(f'graphics/impact_map_co2_vs_el_price_{prefix}.pdf')


#%%########### Plots for suplemental materials #############
############################################################

"""
Boxplot of usage of the allocated CO2 targets for all model nations
"""
def plot_unused_co2():

    df = dfs['df_co2_assigned'].copy()
    df = dfs['df_co2']/df

    df = (df[df.mean().sort_values(ascending=False).index])*100
    df = df.iloc[:,:33]
    df['year'] = dfs['df_chain']['year']

    optimal_index = dfs['df_chain'].query('year == "scenarios_2030_f"').index
    df_optimal = df.iloc[optimal_index]
    df_optimal['Scenario'] = np.array(scenarios)

    df_optimal = df_optimal.query('Scenario != "Efficiency"')


    # filter out burnin samples
    df = df[filt_burnin & filt_co2_cap]

    index_order = list(df.mean().sort_values(ascending=False).index)
    index_order.append('year')

    df = df[index_order]

    plot_box(df,df_optimal,
                ylabel='Fraction of national target used\n[%]',
                prefix=prefix,
                name='unused_co2',
                save=True,
                fliersize=0.1,
                linewidth=0.5,
                color='#bdbdbd')

plot_unused_co2()

#%%
"""
Scatter plots of CO2 target versus the fraction of the target utilized
"""
def plot_country_co2_vs_elec_price(countries):
    
    f,ax = plt.subplots(1,5,sharey=True,figsize=(12,3))

    for i,country in enumerate(countries):
        x = dfs['df_co2_assigned'][country]*1e-6
        x = x[filt_burnin & filt_co2_cap]

        y = (dfs['df_co2'][country].iloc[:-3]/dfs['df_co2_assigned'][country])*100
        y = y[filt_burnin & filt_co2_cap]

        ax[i].scatter(x,y,
                        alpha=0.1,
                        #c='#8f897b',
                        #c='#5c5c5c'
                        s=0.2,
                        c='k'
                        )
        ax[i].set_xticks(np.arange(0, max(x)+1, 20))

        ax[i].set_xlabel('CO$_2$ target\n [M ton CO2]')
        ax[i].set_title(country)
    ax[0].set_ylabel('Fraction of national\ntarget used [%]')
    plt.savefig(f'graphics/co2_vs_co2_{countries}.jpeg',dpi=400,bbox_inches='tight')

plot_country_co2_vs_elec_price(['PL','NL','AT','FI','SE'])

#%% Plot scenario realised emission vs assigned 
"""
Lineplot of the realized emissions per assigned emission for 
all scenarios swept across muliple CO2 budgets
"""

def plot_scenario_emissions_sweep():

    scenarios_index = dfs['df_chain'].query('year == "scenarios_sweep_2030_f" | year == "scenarios_2030_f"').index

    df = pd.DataFrame()
    df['co2_emitted'] =dfs['df_secondary']['co2_emission'].iloc[scenarios_index]
    df['co2_assigned']  = (1-dfs['df_chain'].iloc[scenarios_index].s*1e-2)*base_emis
    # Scale data
    df['co2_assigned'] = df['co2_assigned']*1e-6
    df['co2_emitted'] = df['co2_emitted']*1e-6

    df['Scenario'] = [ivd[c] for c in  dfs['df_chain'].iloc[scenarios_index].c]

    sns.lineplot(data=df,y='co2_emitted',x='co2_assigned',hue='Scenario')
    plt.hlines(base_emis*0.45*1e-6,0.3*1e3,1.4*1e3,colors='k')
    plt.ylabel('Realized emissions\n[M ton CO$_2$]')
    plt.xlabel('Assigned emissions\n[M ton CO$_2$]')
    plt.savefig(f'graphics/co2_realized_vs_assigned_{prefix}.jpeg',dpi=400,bbox_inches='tight')

plot_scenario_emissions_sweep()

#%% Plot national co2 reduction vs co2 reduction cost
"""
Scatterplot of CO2 emissions vs CO2 abatement costs 
"""
def plot_country_red_vs_co2_price(countries,ncols,nrows,figsize=(8,5.5)):
    f,ax = plt.subplots(ncols,nrows,
                        sharey=True,sharex=True,
                        figsize=figsize,
                        )
    ax = ax.flatten()

    for i,country in enumerate(countries):
        df_nodal_co2_reduct = dfs['df_nodal_co2_reduct']*100
        x = df_nodal_co2_reduct[country]
        y =dfs['df_nodal_co2_price'][country]
        ax[i].scatter(y=y,x=x,alpha=0.5,s=0.04,c='black' )
        agg_param = 3
        sns.lineplot(x=(x/agg_param).astype(int)*agg_param+agg_param/2,
                    y=y,
                    ax=ax[i],
                    estimator='mean',
                    ci=0,
                    linewidth=2,
                    color='red',
                    label=None)

        ax[i].set_xlim(-3,105)
        c_name = pytz.country_names[country]
        ax[i].set_title(f'{c_name} ({country})')
        ax[i].set_ylabel(r'CO$_2$ abatement'+'\n cost [€/ton]')
        ax[i].set_xlabel('')
        ax[i].grid()

    ax[0].set_ylabel(r'CO$_2$ abatement'+'\n cost [€/ton]')
    ax[3].set_ylabel(r'CO$_2$ abatement'+'\n cost [€/ton]')
    ax[6].set_ylabel(r'CO$_2$ abatement'+'\n cost [€/ton]')

    plt.ylim(-2,150)
    plt.xticks(np.arange(0, 101, 20))
    plt.yticks(np.arange(0, 151, 50))
    plt.tight_layout()
    
    f.text(0.5, -0.0, r'% CO$_2$ emissions relative to 1990 values', ha='center')

    if len(countries) > 30:
        countries = 'all'

    plt.savefig(f'graphics/red_vs_co2_price_{countries}.jpeg',dpi=150,bbox_inches='tight')



countries = ['DE','DK','ES','GB','IT','NL','PL','RO','SI']
plot_country_red_vs_co2_price(countries,3,3,figsize=(8,5.5))

countries = network.buses.country.unique()
countries = countries[:-1]
countries =  countries[countries !='ME']
plot_country_red_vs_co2_price(countries,8,4,figsize=(14,20))


#%% Renewable capacites versus co2 red
"""
Scatter plot of renewable capacites for all model nations
"""

def plot_scatter_renewable_cap():
    f,ax = plt.subplots(8,4,sharey=False,sharex=True,
                        #figsize=(8,5.5),
                        figsize=(8,10)
                        )
    ax = ax.flatten()

    countries = network.buses.country.unique()
    countries = countries[:-1]
    countries =  countries[countries !='ME']

    for i,country in enumerate(countries):
        df_nodal_gen_p = dfs['df_gen_p'].groupby([network.generators.country,network.generators.carrier],axis=1).sum()

        df_nodal_link_p = dfs['df_links'].loc[:,network.links.query('p_nom_extendable==True').index]
        df_nodal_link_p = df_nodal_link_p.groupby([network.links.country,network.links.carrier],axis=1).sum()

        df_nodal_store_p = dfs['df_store_P'].groupby([network.stores.country,network.stores.carrier],axis=1).sum()

        df_nodal_co2_reduct = dfs['df_nodal_co2_reduct']*100
        x = df_nodal_co2_reduct[country]

        y =df_nodal_gen_p[country]
        y = y.groupby({'offwind-ac':'wind','offwind-dc':'wind','offwind':'wind','onwind':'wind','solar':'solar'},axis=1).sum()
        #y =df_nodal_link_p[country]
        #y= df_nodal_store_p[country]

        colors={'offwind-ac':'blue','offwind-dc':'darkblue','offwind':'m','onwind':'lightblue','ror':'red','solar':'yellow'}

        for c in y.columns:
            ax[i].scatter(y=y[c],x=x,alpha=0.5,s=0.02,
                            c=get_tech_colors()[c],label=c
                            )

            if max(y.max())<100:
                ax[i].set_ylim(0,100)
            elif max(y.max())>1e6:
                ax[i].set_ylim(0,1e6)

        
        if i%4==0:
            ax[i].set_ylabel('Renewables\n[GW]')


        ax[i].set_xlim(-3,105)
        c_name = pytz.country_names[country]
        ax[i].set_title(f'{c_name} ({country})')
        ax[i].set_xlabel('')
        ax[i].grid()

    #ax[-4].legend(bbox_to_anchor=(3, -0.1),ncol=4,markerscale=50)

    plt.xticks(np.arange(0, 101, 20))

    plt.tight_layout()

    f.text(0.5, -0.01, r'% CO$_2$ emissions relative to 1990 values', ha='center')

    plt.savefig(f'graphics/red_vs_renewables_all.jpeg',dpi=150,bbox_inches='tight')

plot_scatter_renewable_cap()


#%%  Storage capacites versus co2 red
"""
Scatter plot of storage capacities for all model countries 
"""

def plot_scatter_storage_cap():

    f,ax = plt.subplots(8,4,sharey=False,sharex=True,
                        #figsize=(8,5.5),
                        figsize=(8,10)
                        )
    ax = ax.flatten()

    countries = network.buses.country.unique()
    countries = countries[:-1]
    countries =  countries[countries !='ME']

    for i,country in enumerate(countries):

        df_nodal_gen_p = dfs['df_gen_p'].groupby([network.generators.country,network.generators.carrier],axis=1).sum()

        df_nodal_link_p = dfs['df_links'].loc[:,network.links.query('p_nom_extendable==True').index]
        df_nodal_link_p = df_nodal_link_p.groupby([network.links.country,network.links.carrier],axis=1).sum()

        df_nodal_store_p = dfs['df_store_P'].groupby([network.stores.country,network.stores.carrier],axis=1).sum()

        df_nodal_co2_reduct = dfs['df_nodal_co2_reduct']*100
        x = df_nodal_co2_reduct[country]

        y= df_nodal_store_p[country]

        colors={'offwind-ac':'blue','offwind-dc':'darkblue','offwind':'m','onwind':'lightblue','ror':'red','solar':'yellow'}

        for c in y.columns:
            ax[i].scatter(y=y[c],x=x,alpha=0.5,s=0.02,
                            c=get_tech_colors()[c],label=c
                            )

            if max(y.max())<100:
                ax[i].set_ylim(0,100)
            elif max(y.max())>1e6:
                ax[i].set_ylim(0,9e5)

        
        if i%4==0:
            ax[i].set_ylabel('Storage\n[GWh]')


        ax[i].set_xlim(-3,105)
        c_name = pytz.country_names[country]
        ax[i].set_title(f'{c_name} ({country})')
        ax[i].set_xlabel('')
        ax[i].grid()

    #ax[-4].legend(bbox_to_anchor=(3, -0.1),ncol=4,markerscale=50)

    plt.xticks(np.arange(0, 101, 20))

    plt.tight_layout()

    f.text(0.5, -0.01, r'% CO$_2$ emissions relative to 1990 values', ha='center')

    plt.savefig(f'graphics/red_vs_storage_all.jpeg',dpi=400,bbox_inches='tight')

plot_scatter_storage_cap()


#%%  OCGT capacites versus co2 red
"""
Scatter plot of OCGT capacity for all model countries
"""
def plot_scatter_OCGT_cap():
    f,ax = plt.subplots(8,4,sharey=False,sharex=True,
                        #figsize=(8,5.5),
                        figsize=(8,10)
                        )
    ax = ax.flatten()

    countries = network.buses.country.unique()
    countries = countries[:-1]
    countries =  countries[countries !='ME']
    for i,country in enumerate(countries):
        df_nodal_link_p = dfs['df_links'].loc[:,network.links.query('p_nom_extendable==True').index]
        df_nodal_link_p = df_nodal_link_p.groupby([network.links.country,network.links.carrier],axis=1).sum()

        df_nodal_co2_reduct = dfs['df_nodal_co2_reduct']*100
        x = df_nodal_co2_reduct[country]
        y =df_nodal_link_p[country]
        for c in ['OCGT']:
            ax[i].scatter(y=y[c],x=x,alpha=0.5,s=0.02,
                            c=get_tech_colors()[c],label=c
                            )
            ax[i].ticklabel_format(useOffset=False)
            if max(y.max())<100:
                ax[i].set_ylim(0,100)
            elif max(y.max())>1e6:
                ax[i].set_ylim(0,9e5)
            else : 
                ax[i].set_ylim(0,max(y.max())*1.05)

        
        if i%4==0:
            ax[i].set_ylabel('OCGT\n[GW]')

        ax[i].set_xlim(-3,105)
        c_name = pytz.country_names[country]
        ax[i].set_title(f'{c_name} ({country})')
        ax[i].set_xlabel('')
        ax[i].grid()

    plt.xticks(np.arange(0, 101, 20))

    plt.tight_layout()

    f.text(0.5, -0.01, r'% CO$_2$ emissions relative to 1990 values', ha='center')

    plt.savefig(f'graphics/red_vs_OCGT_all.jpeg',dpi=400,bbox_inches='tight')

plot_scatter_OCGT_cap()

# %% ################ Unused plot ###################
#####################################################
"""
Boxplot of nations CO2 reduction relative to 1990 values 
"""

def plot_co2_reduction_box():
    df = dfs['df_nodal_co2_reduct'].copy()*100

    df = df[df.median().sort_values().index]
    df['year'] = dfs['df_chain']['year']
    
    optimal_index = dfs['df_chain'].query('year == "scenarios_2030_f"').index
    df_optimal = df.iloc[optimal_index]
    df_optimal['Scenario'] = np.array(scenarios)



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
                           color='#bdbdbd'
                           )

plot_co2_reduction_box()

#%%
"""
Boxplot of nations absolute CO2 emissions
"""

def plot_co2_box():

    #df = dfs['df_co2']/dfs['df_country_energy']
    df = dfs['df_co2']
    df['year'] = dfs['df_chain'][['year']]

    # assign new sort order 
    #df = df[df.mean().sort_values().index]
    df['year'] = dfs['df_chain']['year']
    df = df[index_order_co2mwh]
    

    df_optimal = df.iloc[optimal_index]
    df_optimal['Scenario'] = np.array(scenarios)

    # filter out burnin samples
    df = df[filt_burnin & filt_co2_cap]

    plot_box(df,df_optimal,title='Country co2 emmission',save=True,prefix=prefix,name='co2_box')

plot_co2_box()

#%%
"""
Boxplot of allocated CO2 emissions 
"""
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
                    marker='X',
                    )
    if plot_optimum:
        sns_plot.map_offdiag(plot_lower)

    plt.suptitle(title)

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
                        marker='X',
                        )

        sns_plot.map_offdiag(plot_lower)

    plt.suptitle(title)

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
        optimal_index = dfs['df_chain'].query('year == "scenarios"').index
        optimal_index = optimal_index.append(dfs['df_chain'].query('s == 1 & c == 1').index)
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
                        marker='X',
                        )

        sns_plot.map_offdiag(plot_lower)

    plt.suptitle(title)

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



#%% Correlation of nodal abatement costs

#corr = dfs['df_co2'][dfs['df_chain'].s>burnin_samples].corr()
corr = dfs['df_nodal_co2_price'][dfs['df_chain'].s>burnin_samples].corr()

corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
corr.loc[corr.query('x == y').index,'value'] = 0
heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value']
)

#plt.savefig(f'graphics/co2_cost_correlations_{prefix}.pdf',transparent=True,dpi=400)


#%% Map plot of noral abatement cost correlation 

corr = dfs['df_nodal_co2_price'][dfs['df_chain'].s>burnin_samples].corr()
correlation_map(corr)

#plt.savefig(f'graphics/co2_cost_correlation_map_{prefix}.pdf',transparent=True,dpi=400)


#%% Chain convergence check 

def plot_chain_development():

    fig,ax = plt.subplots(4,3,sharey=True,sharex=True,figsize=(10,10))
    ax = ax.flatten()

    df = dfs['df_theta'].copy()
    df['index'] = df.index
    df['s'] = dfs['df_chain']['s']
    df['c'] = dfs['df_chain']['c']

    df = df.rename(columns=lambda s : 'value '+s if len(s)==2  else s)
    df.pop('value EU')

    for i in range(10):

        df_i = df.query(f'c == 3 & s>{i}').sort_values('s')

        df_i_mean = df_i.expanding(1).mean()
        df_i_mean.s = df_i.s

        df_i_mean.plot(x='s',y=df.columns[:-3],alpha=0.5,legend=False,ax=ax[i])
        ax[i].set_title(f'Chain {i}')

    ax[10].axis('off')
    ax[11].axis('off')
    ax[9].legend(loc=(1.1,-0.3),ncol=4)


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
