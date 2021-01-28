#%%

import pypsa
from _helpers import configure_logging
import numpy as np
from pypsa.linopt import get_var, define_constraints, linexpr


#%%


if 'snakemake' not in globals():
    from _helpers import mock_snakemake
    try:
        snakemake = mock_snakemake('initialize_networks')
    except :
        import os
        os.chdir('..')
        snakemake = mock_snakemake('initialize_networks')
        #os.chdir('..')
    
configure_logging(snakemake)

network = pypsa.Network(snakemake.input.network)
#network.snapshot_weightings *= 8760/len(network.snapshots)

# %%

network.lopf(**snakemake.config.get('solver'))


#%%
solver = {
  "solver_name": 'gurobi',
  "formulation": 'kirchhoff',
  "pyomo": False,
  "solver_options": {
    "threads": 4,
    "method": 2, # barrier
    "crossover": 0,
    "BarConvTol": 1.e-9,
    "FeasibilityTol": 1.e-6,
    "AggFill": 0,
    "PreDual": 0,
    }}

# %%


def extra_functionality(network, snapshots,local_emis ):
    # Local emisons constraints 
    for i, bus in enumerate(network.buses.index):
        vars = []
        constants = []
        for t in network.snapshots: 
            for gen in network.generators.query("bus == '{}' ".format(bus)).index:
                vars.append(get_var(network,'Generator','p').loc[t,gen])
                const = 1/network.generators.efficiency.loc[gen] 
                const *= network.snapshot_weightings.loc[t]
                const *= network.carriers.co2_emissions.loc[network.generators.carrier.loc[gen]]

                constants.append(const)

        expr = linexpr((constants,vars)).sum()
        define_constraints(network,expr,'<=',local_emis[i],'local_co2','bus {}'.format(i))
            

a = np.array(co2_emis)*0.4

extra_func = lambda n, s: extra_functionality(n, 
                                            s, 
                                            a, )

stat = network.lopf(**solver,keep_shadowprices=True,
                    extra_functionality=extra_func)

#%%
network.generators.query("bus == 'PL0 0'").p_nom_opt
#%%

BASE_COST = 269683914613.72296


print('price increase ',network.objective-base_cost)

global_co2_price = network.duals.co2_con.df.loc[0,'1']
print('global co2 price ',global_co2_price)


print('local co2 price' ,network.duals.local_co2.df.loc[0,'bus 30'])


co2_emis = {}
for bus in network.buses.index:
    local_emis = 0 
    for gen in network.generators.query("bus == '{}' ".format(bus)).index:

        gen_emis = 1/network.generators.efficiency.loc[gen] 
        gen_emis *= (network.snapshot_weightings*network.generators_t.p.T).T.sum().loc[gen]
        try:
            gen_emis *= network.carriers.co2_emissions.loc[network.generators.carrier.loc[gen]]
        except :
            pass
        local_emis += gen_emis
    co2_emis[bus] = local_emis

print('total emissions',sum(co2_emis))


#%%

def draw_sample(theta,network):
    extra_func = lambda n, s: extra_functionality(n, 
                                            s, 
                                            a, )

    stat = network.lopf(**solver,keep_shadowprices=True,
                        extra_functionality=extra_func)


# %%

from iso3166 import countries
import plotly.graph_objects as go 

fig = go.Figure()


#codes = ['AUT', 'BEL', 'BGR', 'BIH', 'HRV', 'CHE', 'CZE', 'DNK', 'EST', 'FIN', 'FRA', 'DEU', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'NLD', 'NOR', 'POL', 'PRT', 'ROU', 'SRB','SVK', 'SVN', 'ESP', 'SWE', 'GBR']
codes = [countries.get(alpha2[:2]).alpha3 for alpha2 in network.buses.index]
names = [countries.get(alpha2[:2]).name for alpha2 in network.buses.index]
#values = network.generators.p_nom_max[network.generators.carrier == 'onwind']


contry_emissions = {code : 0 for code in codes}
for i,emis in enumerate(co2_emis):
    contry_emissions[codes[i]] += emis



fig.add_trace(go.Choropleth(
                    geo='geo1',
                    locations = list(contry_emissions.keys()),
                    z = list(contry_emissions.values()),#/area,
                    text = list(contry_emissions.keys()),
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

dir(network)
# %%
# %%
