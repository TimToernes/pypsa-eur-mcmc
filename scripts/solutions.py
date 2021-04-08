#%%
import pandas as pd
import numpy as np
from multiprocessing import Queue
from _mcmc_helpers import str_to_theta

#%% Solutions class
class solutions:
    # the solutions class contains all nececary data for all MGA solutions
    # The class also contains functions to append new solutions and to save the results

    def __init__(self,network,manager=None):
        self.old_objective = network.objective
        self.sum_vars = self.calc_sum_vars(network)

        self.gen_p =    pd.DataFrame(data=[network.generators.p_nom_opt],index=[0])
        self.gen_E =    pd.DataFrame(data=[network.generators_t.p.sum()],index=[0])

        self.storeage_unit_p =  pd.DataFrame(data=[network.storage_units.p_nom_opt],index=[0])
        self.storeage_unit_E =  pd.DataFrame(data=[network.storage_units_t.p.sum()],index=[0])

        self.store_p = pd.DataFrame(data=[network.stores.e_nom_opt],index=[0])
        self.store_E = pd.DataFrame(data=[network.stores_t.p.sum()],index=[0])

        self.links =    pd.DataFrame(data=[network.links.p_nom_opt],index=[0])
        self.lines =    pd.DataFrame(data=[network.lines.s_nom_opt],index=[0])

        try : 
            co2_emis = pd.Series(self.get_country_emis(network))
            self.co2_pr_node = pd.DataFrame(columns=co2_emis.index,data=[co2_emis.values])
        except : 
            self.co2_pr_node = pd.DataFrame()

        try : 
            self.theta = pd.DataFrame([str_to_theta(network.theta)])
        except : 
            self.theta = pd.DataFrame()

        nodal_costs = self.calc_nodal_costs(network)
        self.nodal_costs = pd.DataFrame(columns=nodal_costs.index,data=[nodal_costs.values])

        self.secondary_metrics = self.calc_secondary_metrics(network)
        self.objective = pd.DataFrame()
        try :
            c = network.chain 
            s = network.sample
            a = network.accepted
            self.df_chain = pd.DataFrame(data=dict(c=[c],s=[s],a=[a]))
        except : 
            self.df_chain = pd.DataFrame()

        if manager != None:
            self.queue = manager.Queue()

        self.df_list = ['gen_p', 
                         'gen_E', 
                         'storeage_unit_E', 
                         'storeage_unit_p', 
                         'store_E', 
                         'store_p', 
                         'links', 
                         'lines', 
                         'co2_pr_node', 
                         'sum_vars', 
                         'secondary_metrics',
                         'nodal_costs',
                         'theta',
                         'df_chain']    


    def put(self,network):
    # add new data to the solutions queue. This is used when new data is added from
    # sub-process, when using multiprocessing
        try :
            self.queue
        except:
            print('creating queue object')
            self.queue = Queue()

        part_result = solutions(network)
        self.queue.put(part_result,block=True,timeout=120)

    def init_queue(self):
        # Initialize results queue
        try :
            self.queue.qsize()
        except :
            self.queue = Queue()

    def merge(self):
        # Merge all solutions put into the solutions queue into the solutions dataframes
        #merge_num = self.queue.qsize()
        merge_num='!! not working !!'
        while not self.queue.empty() :
            part_res = self.queue.get(120)

            for df_name in self.df_list:
                part_res_df = getattr(part_res,df_name)
                self.__dict__[df_name] = self.__dict__[df_name].append(part_res_df,ignore_index=True)


    def save_csv(self, file_prefix='sol'):
        # Save a csv file for all dataframes in the df_list
        for df_name in self.df_list:
            self.__dict__[df_name].to_csv(file_prefix+df_name+".csv")


    def calc_secondary_metrics(self,network):
        # Calculate secondary metrics
        gini = self.calc_gini(network)
        gini_co2 = self.calc_co2_gini(network)
        co2_emission = self.calc_co2_emission(network)
        system_cost = self.calc_system_cost(network)
        autoarky = self.calc_autoarky(network)
        return pd.DataFrame({'system_cost':system_cost,'co2_emission':co2_emission,'gini':gini,'gini_co2':gini_co2,'autoarky':autoarky},index=[0])

    def calc_sum_vars(self,network):
        sum_data = dict(network.generators.p_nom_opt.groupby(network.generators.carrier).sum())
        sum_data['transmission'] = network.links.p_nom_opt.sum()
        sum_data['co2_emission'] = self.calc_co2_emission(network)
        sum_data.update(network.storage_units.p_nom_opt.groupby(network.storage_units.carrier).sum())
        sum_vars = pd.DataFrame(sum_data,index=[0])
        return sum_vars

    def calc_gini(self,network):
    # This function calculates the gini coefficient of a given PyPSA network.

        bus_total_prod = network.generators_t.p.sum().groupby(network.generators.location).sum()

        ac_buses = network.buses.query('carrier == "AC"').index
        generator_link_carriers = ['OCGT', 'CCGT', 'coal', 'lignite', 'nuclear', 'oil']
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

        return gini

    def calc_autoarky(self,network):
        # calculates the autoarky of a model solution 
        # autoarky is calculated as the mean self sufficiency (energy produced/energy consumed) of all countries in all hours
        mean_autoarky = []
        for snap in network.snapshots:
            hourly_load = network.loads_t.p_set.loc[snap]
            hourly_autoarky = network.generators_t.p.loc[snap].groupby(network.generators.bus).sum()/hourly_load
            hourly_autoarky_corected = hourly_autoarky.where(hourly_autoarky<1,1)
            mean_autoarky.append(np.mean(hourly_autoarky_corected))
        return np.mean(mean_autoarky)

    def calc_co2_emission(self,network):
            #CO2

        w_t =  network.snapshot_weightings
        n_ns = network.generators.efficiency
        g_nst = network.generators_t.p
        e_ns = network.carriers.co2_emissions

        co2_emission = g_nst.mul(w_t,axis='rows')
        co2_emission = co2_emission.mul(1/n_ns,axis='columns')
        co2_emission = co2_emission.sum().groupby(network.generators.carrier).sum()
        co2_emission = co2_emission * e_ns
        co2_emission = co2_emission.fillna(0).sum()
        
        #id_ocgt = network.generators.index[network.generators.carrier == 'ocgt']
        #co2_emission = network.generators_t.p[id_ocgt].sum().sum()*network.carriers.co2_emissions['ocgt']/network.generators.efficiency.iloc[0]
        #co2_emission
        return co2_emission

    def calc_system_cost(self,network):
        #Cost
        #capital_cost = sum(network.generators.p_nom_opt*network.generators.capital_cost) + sum(network.links.p_nom_opt*network.links.capital_cost) + sum(network.storage_units.p_nom_opt * network.storage_units.capital_cost)
        #marginal_cost = network.generators_t.p.groupby(network.generators.carrier,axis=1).sum().sum() * network.generators.marginal_cost.groupby(network.generators.type).mean()
        #total_system_cost = marginal_cost.sum() + capital_cost
        total_system_cost = network.objective
        return total_system_cost


    def get_country_emis(self,network):

        query_string = lambda x : f'bus0 == "{x}" | bus1 == "{x}" | bus2 == "{x}" | bus3 == "{x}" | bus4 == "{x}"'
        id_co2_links = network.links.query(query_string('co2 atmosphere')).index

        country_codes = network.links.loc[id_co2_links].location.unique()
        country_emis = {code:0 for code in country_codes}

        for country in country_codes:
            idx = network.links.query(f'location == "{country}"').index
            id0 = (network.links.loc[idx] == 'co2 atmosphere')['bus0']
            country_emis[country] -= network.links_t.p0[idx[id0]].sum(axis=1).mul(network.snapshot_weightings).sum()
            id1 = (network.links.loc[idx] == 'co2 atmosphere')['bus1']
            country_emis[country] -= network.links_t.p1[idx[id1]].sum(axis=1).mul(network.snapshot_weightings).sum()
            id2 = (network.links.loc[idx] == 'co2 atmosphere')['bus2']
            country_emis[country] -= network.links_t.p2[idx[id2]].sum(axis=1).mul(network.snapshot_weightings).sum()
            id3 = (network.links.loc[idx] == 'co2 atmosphere')['bus3']
            country_emis[country] -= network.links_t.p3[idx[id3]].sum(axis=1).mul(network.snapshot_weightings).sum()
            id4 = (network.links.loc[idx] == 'co2 atmosphere')['bus4']
            country_emis[country] -= network.links_t.p4[idx[id4]].sum(axis=1).mul(network.snapshot_weightings).sum()

            if country == 'EU':
                id_load_co2 = network.loads.query('bus == "co2 atmosphere"').index
                co2_load = network.loads.p_set[id_load_co2].sum().sum()*sum(network.snapshot_weightings)
                country_emis[country] -= co2_load

            total_emis = np.sum(list(country_emis.values())) 
        
        return country_emis


    def calc_co2_emis_pr_node(self,network):

        co2_emis = {}
        for bus in network.buses.index:
            local_emis = 0 
            for gen in network.generators.query("bus == '{}' ".format(bus)).index:

                gen_emis = 1/network.generators.efficiency.loc[gen] 
                gen_emis *= (network.snapshot_weightings*network.generators_t.p.T).T.sum().loc[gen]
                try:
                    gen_emis *= network.carriers.co2_emissions.loc[network.generators.carrier.loc[gen]]
                except Exception:
                    pass
                local_emis += gen_emis
            co2_emis[bus] = local_emis
            co2_emis = pd.Series(co2_emis)
        return co2_emis



    def calc_co2_gini(self,network):

        co2_emis = self.calc_co2_emis_pr_node(network)
        co2_emis = np.array(self.co2_pr_node)[0]
        #co2_emis = pd.Series(co2_emis)

        #bus_total_prod = network.generators_t.p.sum().groupby(network.generators.bus).sum()
        load_total= network.loads_t.p_set.sum()
        load_total = load_total.groupby(network.buses.country).sum()
        try : 
            load_total.pop('')
        except : 
            pass 

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



    def calc_nodal_costs(self,network):

        def _calculate_nodal_costs(n,nodal_costs):
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
        nodal_costs = _calculate_nodal_costs(network,nodal_costs)

        return nodal_costs 


# %%

# testing of the class 

if __name__ == '__main__':
    import pypsa
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


    network = pypsa.Network('results/mcmc_2030_H/network_c0_s1.nc',
                            override_component_attrs=override_component_attrs)

    sol = solutions(network)

    network = set_link_locataions(network)
    
    sol.put(network)

    sol.merge()
# %%
