#%%
import pypsa
import os 
import csv
import numpy as np 
import re 
from _helpers import configure_logging
import multiprocessing as mp
import queue # imported for using queue.Empty exception
import time 
#%%

def write_csv(path,item):
    # Write a list or numpy array (item) as csv file 
    # to the specified path
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(item)

def read_csv(path):
    item = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            item.append(row)
    return item

def increment_sample(path,incr=1):
    folder,file = os.path.split(path)
    sample_str = re.search('s([0-9]?[0-9]?[0-9]?[0-9])',file).group()
    current_sample = int(sample_str[1:])
    file_new = file.replace(sample_str,'s{}'.format(current_sample+incr))
    path_out = os.path.join(folder, file_new )
    return path_out

def get_theta(network,mcmc_variables):
    theta = []
    for var in mcmc_variables:
        if network.generators.index.isin(var).any():
            theta.append(network.generators.p_nom_opt.loc[network.generators.index.isin(var)].sum())
        else :
            theta.append(network.storage_units.p_nom_opt.loc[network.storage_units.index.isin(var)].sum())

    return theta

def worker(q,thetas,mcmc_variables,q_proc_done):
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
                network = pypsa.Network('inter_results/'+file)
                theta = get_theta(network,mcmc_variables)
                thetas.put(theta)
                del(network)

#%%
if __name__=='__main__':
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        try:
            snakemake = mock_snakemake('calc_sigma',sample=101)
            os.chdir('..')
        except :
            os.chdir(os.getcwd()+'/scripts')
            snakemake = mock_snakemake('calc_sigma',sample=101)
            os.chdir('..')
            
    configure_logging(snakemake,skip_handlers=True)

    # impor a network 
    network = pypsa.Network(snakemake.input[0])
    mcmc_variables = read_csv(network.mcmc_variables)
    eps = snakemake.config['sampler']['eps']

    man = mp.Manager()


    thetas_q = man.Queue()
    # Itterate over all files in the inter_results directory 
    # If the file is a network file, it is loaded and the theta value is extracted and stored to the thetas_q list
 
    dir_lst = os.listdir('inter_results/')
    q = man.Queue()
    q_proc_done = man.Queue()
    for d in dir_lst:
        q.put(d)

    processes = []
    for i in range(snakemake.threads):
        p = mp.Process(target=worker,args=(q,thetas_q,mcmc_variables,q_proc_done))
        p.start()
        processes.append(p)
    
    while q_proc_done.qsize()<snakemake.threads:
        time.sleep(1)


    for p in processes:
        p.kill()
        p.join()

    thetas = []
    while thetas_q.qsize()>0:
        thetas.append(thetas_q.get(30))

    thetas = np.array(thetas)

    # Calculate sigma from the data. 
    # If the sigma values are low, the eps parameter is added to the diagonal 
    sigma = np.cov(thetas.T) 
    if np.mean(sigma.diagonal())<eps:
        sigma += np.identity(thetas.shape[1])*eps

    # Save sigme as csv and update the network.sigma path in the most recent networks
    np.savetxt(snakemake.output[0],sigma)

    for inp in snakemake.input:
        network = pypsa.Network(inp)
        network.sigma = increment_sample(network.sigma,100)
        network.export_to_netcdf(inp)


# %%
