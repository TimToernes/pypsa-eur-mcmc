#%%
import pandas as pd
import numpy as np
from multiprocessing import Queue

#%% Solutions class
class solutions:
    # the solutions class contains all nececary data for all MGA solutions
    # The class also contains functions to append new solutions and to save the results

    def __init__(self,network,manager=None):
        self.old_objective = network.objective
        self.sum_vars = self.calc_sum_vars(network)
        self.gen_p =    pd.DataFrame(data=[network.generators.p_nom_opt],index=[0])
        self.gen_E =    pd.DataFrame(data=[network.generators_t.p.sum()],index=[0])
        self.store_p =  pd.DataFrame(data=[network.storage_units.p_nom_opt],index=[0])
        self.store_E =  pd.DataFrame(data=[network.storage_units_t.p.sum()],index=[0])
        self.links =    pd.DataFrame(data=[network.links.p_nom_opt],index=[0])
        self.lines =    pd.DataFrame(data=[network.lines.s_nom_opt],index=[0])
        co2_emis = self.calc_co2_emis_pr_node(network)
        self.co2_pr_node = pd.DataFrame(columns=co2_emis.index,data=[co2_emis.values])
        self.secondary_metrics = self.calc_secondary_metrics(network)
        self.objective = pd.DataFrame()
        try :
            c = network.chain 
            s = network.sample
            self.df_chain = pd.DataFrame(data=dict(c=[c],s=[s]))
        except : 
            pass 

        if manager != None:
            self.queue = manager.Queue()

        self.df_list = {'gen_p':self.gen_p,
                        'gen_E':self.gen_E,
                        'store_E':self.store_E,
                        'store_p':self.store_p,
                        'links':self.links,
                        'lines':self.lines,
                        'co2_pr_node':self.co2_pr_node,
                        'sum_vars':self.sum_vars,
                        'secondary_metrics':self.secondary_metrics}

        #try :
        #    co2_emission = [constraint.body() for constraint in network.model.global_constraints.values()][0]
        #except :
        #    co2_emission = 0

    def append(self,network):
        # Append new data to all dataframes
        self.sum_vars = self.sum_vars.append(self.calc_sum_vars(network),ignore_index=True)
        self.gen_p =    self.gen_p.append([network.generators.p_nom_opt],ignore_index=True)
        self.links =    self.gen_p.append([network.links.p_nom_opt],ignore_index=True)
        self.gen_E =    self.gen_E.append([network.generators_t.p.sum()],ignore_index=True)
        self.secondary_metrics = self.secondary_metrics.append(self.calc_secondary_metrics(network),ignore_index=True)

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
            part_res = self.queue.get(60)
            self.gen_E = self.gen_E.append(part_res.gen_E,ignore_index=True)
            self.gen_p = self.gen_p.append(part_res.gen_p,ignore_index=True)
            self.store_E = self.store_E.append(part_res.store_E,ignore_index=True)
            self.store_p = self.store_p.append(part_res.store_p,ignore_index=True)
            self.links = self.links.append(part_res.links,ignore_index=True)
            self.lines = self.lines.append(part_res.lines,ignore_index=True)
            self.co2_pr_node = self.co2_pr_node.append(part_res.co2_pr_node,ignore_index=True)
            self.sum_vars = self.sum_vars.append(part_res.sum_vars,ignore_index=True)
            self.secondary_metrics = self.secondary_metrics.append(part_res.secondary_metrics,ignore_index=True)
            try : 
                self.df_chain
            except : 
                pass
            else : 
                self.df_chain = self.df_chain.append(part_res.df_chain,ignore_index=True)
        #print('merged {} solution'.format(merge_num))

    def save_xlsx(self,file='save.xlsx'):
        # Store all dataframes als excel file
        self.df_list = {'gen_p':self.gen_p,
                'gen_E':self.gen_E,
                'store_E':self.store_E,
                'store_p':self.store_p,
                'links':self.links,
                'lines':self.lines,
                'sum_vars':self.sum_vars,
                'secondary_metrics':self.secondary_metrics}

        writer = pd.ExcelWriter(file)
        #sheet_names =  ['gen_p','gen_E','links','sum_var','secondary_metrics']
        for i, df in enumerate(self.df_list):
            self.df_list[df].to_excel(writer,df)
        writer.save()
        print('saved {}'.format(file))

    def save_csv(self, file_prefix='sol'):
        self.df_list = {'gen_p':self.gen_p,
                'gen_E':self.gen_E,
                'store_E':self.store_E,
                'store_p':self.store_p,
                'links':self.links,
                'lines':self.lines,
                'co2_pr_node':self.co2_pr_node,
                'sum_vars':self.sum_vars,
                'secondary_metrics':self.secondary_metrics}
        try : 
            self.df_chain
        except : 
            pass
        else : 
            self.df_list['chain'] = self.df_chain
        
        for i, df in enumerate(self.df_list):
            self.df_list[df].to_csv(file_prefix+df+".csv")
        


    def calc_gini(self,network):
    # This function calculates the gini coefficient of a given PyPSA network.
        bus_total_prod = network.generators_t.p.sum().groupby(network.generators.bus).sum()
        load_total= network.loads_t.p_set.sum()

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
        #co2_emis = pd.Series(co2_emis)

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


# %%
