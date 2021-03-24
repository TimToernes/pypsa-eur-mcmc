# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd


# %%
def draw_theta(theta,eps,upper_bound=1,lower_bound=0,):
    def unif(t,s_lb,s_ub):
        if np.random.rand()>0.5:
            s = np.random.uniform(s_lb,t)
        else :
            s = np.random.uniform(t,s_ub)
        return s

    if type(upper_bound) == int:
        upper_bound = np.ones(len(theta))*upper_bound
    if type(lower_bound) == int:
        lower_bound = np.ones(len(theta))*lower_bound
    

    theta_proposed = np.zeros(len(theta))
    for i,t in enumerate(theta): 
        lower = max([t-eps[i]/2,lower_bound[i]])
        upper = min([t+eps[i]/2,upper_bound[i]])
        theta_proposed[i] = np.random.uniform(lower,upper,1)
    #if sum(theta_proposed)>1:
    scale_lb = max([sum(theta)-np.mean(eps),0])
    scale_ub = min([sum(theta)+np.mean(eps),1])
    theta_proposed = theta_proposed/sum(theta_proposed)*unif(sum(theta),scale_lb,scale_ub)
    return theta_proposed


# %%
dim=30
theta = np.random.rand(dim)
theta = theta/sum(theta)*0.1
eps = np.ones(dim)*0.2
#x = np.array([sample(theta,eps) for i in range(3000)])


# %%
thetas = []
for i in range(10000):
    theta = draw_theta(theta,eps) 
    if sum(theta)>0:
        thetas.append(theta)
thetas = np.array(thetas)


# %%
len(thetas)


# %%
fig,ax = plt.subplots(tight_layout=True)
ax.hist(thetas.sum(axis=1),bins=30,)


# %%
fig,ax = plt.subplots(tight_layout=True)
ax.hist(thetas.sum(axis=1))


# %%
plt.figure()
plt.plot(thetas[:50,0],thetas[:50,1])


# %%
import pypsa


# %%
network = pypsa.Network('inter_results/network_c0_s1.nc')


# %%

def extra_functionality(network, snapshots,variables,local_emis ):
    # Local emisons constraints 
    for i, bus in enumerate(variables):
        vars = []
        constants = []
        for t in network.snapshots: 
            for gen in network.generators.query('bus == {}'.format(str(bus))).index:
                vars.append(get_var(network,'Generator','p').loc[t,gen])
                const = 1/network.generators.efficiency.loc[gen] 
                const *= network.snapshot_weightings.loc[t]
                const *= network.carriers.co2_emissions.loc[network.generators.carrier.loc[gen]]

                constants.append(const)

        expr = linexpr((constants,vars)).sum()
        define_constraints(network,expr,'<=',local_emis[i],'local_co2','bus {}'.format(i))


# %%
solver = {
  "solver_name": 'gurobi',
  "formulation": 'kirchhoff',
  "pyomo": False,
  "keep_shadowprices": True,
  "solver_options": {
    "threads": 4,
    "method": 2, # barrier
    "crossover": 0,
    "BarConvTol": 1.e-6,
    "FeasibilityTol": 1.e-6,
    "AggFill": 0,
    "PreDual": 0,
    }}


# %%
from _mcmc_helpers import get_theta,calc_co2_emis_pr_node,read_csv
from pypsa.linopt import get_var, define_constraints, linexpr


# %%
co2_budget = 3875000
network.global_constraints.constant.CO2Limit = co2_budget


# %%
network.lopf(**solver,)
network.objective_optimum = network.objective 


# %%
emis_pr_node = calc_co2_emis_pr_node(network)
plot_map(network,mcmc_variables,pd.Series(emis_pr_node))


# %%
gen_pr_node = network.generators_t.p.sum().groupby(network.generators.bus).sum()
plot_map(network,mcmc_variables,gen_pr_node)


# %%
network.carriers


# %%
p_OCGT_pr_node = network.generators.query('carrier == "onwind"').p_nom_opt.groupby(network.generators.bus).sum()


# %%
plot_map(network,mcmc_variables,p_OCGT_pr_node)


# %%
theta = np.zeros(33) 
theta[0] = 0.5
co2_aloc = co2_budget * theta


# %%
mcmc_variables = read_csv(network.mcmc_variables)
extra_func = lambda n, s: extra_functionality(n,
                                            s,
                                            mcmc_variables,
                                            co2_aloc)

stat = network.lopf(**solver,
                    extra_functionality=extra_func)


# %%
(network.objective-network.objective_optimum)/network.objective_optimum*100


# %%
calc_co2_emis_pr_node(network)


# %%
network.generators.query('carrier == "onwind"').p_nom.sum()


# %%
network.generators.query('carrier == "OCGT"').p_nom.sum()


# %%
c_links = sum((network.links.p_nom_opt - network.links.p_nom)*network.links.capital_cost)
c_gen = sum(network.generators.p_nom_opt*network.generators.capital_cost)
c_store = sum(network.storage_units.p_nom_opt*network.storage_units.capital_cost)
c_marg = sum(network.generators_t.p.sum() * network.generators.marginal_cost)
c_lines = sum((network.lines.s_nom_opt-network.lines.s_nom) * network.lines.capital_cost)


# %%
c_links + c_gen + c_marg + c_store


# %%
network.objective


# %%
(c_links + c_gen + c_marg + c_store + c_lines - network.objective)/network.objective * 100


# %%
network.plot()


# %%
network.generators_t.p_max_pu.sum().groupby(network.generators.bus).sum()


# %%
network.generators_t.p_max_pu


# %%
gen_pr_node


# %%



# %%



# %%
def plot_map(network,mcmc_variables,pr_node_series):
    from iso3166 import countries
    import plotly.graph_objects as go 
    import pypsa
    from _mcmc_helpers import calc_co2_emis_pr_node, read_csv


    agg_series = {}
    for i in range(33):
        alpha2_mcmc_variable = mcmc_variables[i][0][:2]
        alpha3 = countries.get(alpha2_mcmc_variable).alpha3
        try :
            agg_series[alpha3] += sum(pr_node_series [mcmc_variables[i]].values)
        except : 
            try :
                agg_series[alpha3] = sum(pr_node_series [mcmc_variables[i]].values)
            except : 
                agg_series[alpha3] = 0 


    fig = go.Figure()
    fig.add_trace(go.Choropleth(
                        geo='geo1',
                        locations = list(agg_series.keys()),
                        z = list(agg_series.values()),#/area,
                        text = list(agg_series.keys()),
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



