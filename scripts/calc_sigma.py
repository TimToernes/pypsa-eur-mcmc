#%%
import pypsa
import os 
import csv
import numpy as np 
import pandas as pd
import re 
from _helpers import configure_logging
from _mcmc_helpers import *
import multiprocessing as mp
import queue # imported for using queue.Empty exception
import time 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



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

#%%

def new_files(snakemake):
    """ 
    The function will return the a list of the network files 
    that haven't been processed yet
    """
    batch = int(snakemake.config['sampler']['batch'])
    chains = int(snakemake.config['sampler']['chains'])
    sample_n = int(snakemake.wildcards['sample'])

    files = []
    for c in range(chains):
        for s in range(sample_n-batch,sample_n):
            files.append(f'network_c{c}_s{s}.nc')
    return files


def increment_sample(path,incr=1):
    folder,file = os.path.split(path)
    sample_str = re.search('s([0-9]?[0-9]?[0-9]?[0-9])',file).group()
    current_sample = int(sample_str[1:])
    file_new = file.replace(sample_str,'s{}'.format(current_sample+incr))
    path_out = os.path.join(folder, file_new )
    return path_out



def worker(q,thetas,mcmc_variables,co2_budget,q_proc_done,snakemake):
    proc_name = mp.current_process().name
    while True:
        try:
            #try to get task from the queue. get_nowait() function will 
            #raise queue.Empty exception if the queue is empty. 
            #queue(False) function would do the same task also.
            file = q.get(False)
        except queue.Empty:
            print('no more jobs - {}'.format(proc_name))
            q_proc_done.put(proc_name)
            break
        else:
            if file[:7] == 'network':
                network = pypsa.Network(f'inter_results/{snakemake.config["run_name"]}/'+file,override_component_attrs=override_component_attrs)
                theta = {}
                theta['s'] = network.sample
                theta['c'] = network.chain
                theta['a'] = network.accepted
                #theta['val'] = get_theta(network,mcmc_variables,co2_budget)
                #country_emis = get_country_emis(network)
                #theta['val'] = np.array([country_emis[v] for v in mcmc_variables])/co2_budget
                theta['val'] = str_to_theta(network.theta)

                thetas.put(theta)
                del(network)

def schedule_workers(mcmc_variables,co2_budget):
    """
    Input: mcmc_variables
    Output: dataframe containing theta values for unprocessed networks 
    """
    # Start multiprocessing manager and que for thetas 
    man = mp.Manager()
    thetas_q = man.Queue()

    # Itterate over all files in the inter_results directory 
    # If the file is a network file, it is loaded and the theta 
    # value is extracted and stored to the thetas_q list
    dir_lst = new_files(snakemake)
    q = man.Queue()
    q_proc_done = man.Queue()
    for d in dir_lst:
        q.put(d)

    processes = []
    for i in range(snakemake.threads):
        p = mp.Process(target=worker,args=(q,thetas_q,mcmc_variables,co2_budget,q_proc_done,snakemake))
        p.start()
        processes.append(p)
    
    while q_proc_done.qsize()<snakemake.threads:
        time.sleep(1)

    for p in processes:
        p.kill()
        p.join()

    thetas = []
    s_lst = []
    c_lst = []
    a_lst = []
    while thetas_q.qsize()>0:
        theta = thetas_q.get(30)
        thetas.append(theta['val'])
        s_lst.append(theta['s'])
        c_lst.append(theta['c'])
        a_lst.append(theta['a'])


    df = pd.DataFrame(data=thetas,columns=[str(x) for x in range(34)])
    df['s'] = s_lst
    df['c'] = c_lst
    df['a'] = a_lst

    return df

#%%
if __name__=='__main__':
    # Setup of snakemake when debugging 
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        try:
            snakemake = mock_snakemake('calc_sigma',sample=6,run_name='mcmc_2030')
        except :
            os.chdir('..')
            snakemake = mock_snakemake('calc_sigma',sample=6,run_name='mcmc_2030')
    # Setup logging
    configure_logging(snakemake,skip_handlers=True)

    # impor a network and other variables 
    network = pypsa.Network(snakemake.input[0])
    mcmc_variables = read_csv(network.mcmc_variables)
    mcmc_variables = [row[0]+row[1] for row in mcmc_variables]
    eps = float(snakemake.config['sampler']['eps'])

    # Get the theta value for all new networks
    co2_budget = snakemake.config['co2_budget']
    df_theta = schedule_workers(mcmc_variables,co2_budget)

    if os.path.isfile(f'inter_results/{snakemake.config["run_name"]}/theta.csv'):
        df_theta_old = pd.read_csv(f'inter_results/{snakemake.config["run_name"]}/theta.csv',
                                    index_col=0)
        df_theta = pd.concat([df_theta,df_theta_old],ignore_index=True)
    
    df_theta.to_csv(f'inter_results/{snakemake.config["run_name"]}/theta.csv')

    thetas = np.array(df_theta.iloc[:,0:34])

    # Calculate sigma from the data. 
    # If the sigma values are low, the eps parameter is added to the diagonal 
    #sigma = np.cov(thetas.T) 
    sigma = np.std(thetas,axis=0)
    #if np.mean(sigma.diagonal())<eps:
    #sigma += np.identity(thetas.shape[1])*eps
    sigma[sigma<eps] = eps
    #else :
    #    sigma += np.identity(thetas.shape[1])*0.1

    # Save sigme as csv and update the network.sigma path in the most recent networks
    np.savetxt(snakemake.output['sigma'],sigma)
    #np.savetxt(snakemake.output['theta'],thetas)

    for inp in snakemake.input:
        network = pypsa.Network(inp)
        network.sigma = increment_sample(network.sigma,snakemake.config['sampler']['batch'])
        network.export_to_netcdf(inp)


# %%
