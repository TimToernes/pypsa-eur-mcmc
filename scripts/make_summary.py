"""
This script takes care of reading all the networks generated with the Metropolis Hasting sampler
with the "mh_chain.py" script. All networks from the inter_results/ folder is open and 
added to the Solutions object. The Solutions object is responsible for generating summary 
data for all networks. 
"""

#%%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pypsa
import os 
import numpy as np 
from solutions import solutions
import multiprocessing as mp
import queue # imported for using queue.Empty exception
import time 
import pickle

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

def worker(q_in,sol,q_proc_done):
    """
    Worker process that opens a network and adds it to the Solutions object
    input:
    q_in: job queue
    sol: Solutions object
    q_proc_done: que containing to communicate finished worker process
    """
    proc_name = mp.current_process().name
    while True:
        try:
            #try to get task from the queue. get_nowait() function will 
            #raise queue.Empty exception if the queue is empty. 
            #queue(False) function would do the same task also.
            file = q_in.get(False)
        except queue.Empty:
            print('no more jobs - {}'.format(proc_name))
            q_proc_done.put(proc_name)
            break
        else:
            if file[:7] == 'network' and file[-2:] == 'nc':
                network = pypsa.Network(f'inter_results/{snakemake.config["run_name"]}/'+file,override_component_attrs=override_component_attrs)
                network.duals , network.dualvalues = pickle.load( open(network.dual_path, "rb" ) )
                sol.put(network)
                del(network)


#%%
if __name__=='__main__':
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        try:
            snakemake = mock_snakemake('data_postprocess')
            #os.chdir('..')
        except :
            os.chdir('..')
            snakemake = mock_snakemake('data_postprocess')
            #os.chdir('..')
            
    #configure_logging(snakemake,skip_handlers=True)
    import logging
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    dir_lst = os.listdir(f'inter_results/{snakemake.config["run_name"]}/')
    network = pypsa.Network(snakemake.input[0],override_component_attrs=override_component_attrs)
    network.duals , network.dualvalues = pickle.load( open(network.dual_path, "rb" ) )
    man = mp.Manager()
    sol = solutions(network, man)
    q_in = man.Queue()
    q_proc_done = man.Queue()
        
    for file in dir_lst:
        q_in.put(file)

    
    print(f'starting {snakemake.threads-2} jobs')
    processes = []
    for i in range(snakemake.threads-2):
        #logging.info(1)
        p = mp.Process(target=worker,args=(q_in,sol,q_proc_done))
        #logging.info(2)
        p.start()
        #logging.info(3)
        processes.append(p)
        logging.info(f'Started job {i} with name {p.pid}')
    logging.info(f'started {len(processes)} jobs')

    timeout_time = 60*60*3 #3 hours in seconds 

    timer = time.time()
    time_diff = time.time()-timer
    logging.info(f'Waiting for jobs to finish')
    while q_proc_done.qsize()<len(processes) or time_diff<timeout_time:
        time_diff = time.time()-timer
        time.sleep(1)

    if time_diff>timeout_time:
        logging.warning('Timed out')
    
    logging.info(f'marging solutions ')
    sol.merge()

    sol.save_csv(f'results/{snakemake.config["run_name"]}/result_')

    logging.info(f'joining processes')
    for p in processes:
        p.kill()
        p.join()


    #copyfile(f'inter_results/{snakemake.config["run_name"]}/theta.csv',f'results/{snakemake.config["run_name"]}/theta.csv')
    
    #sol.save_xlsx('results/result.xlsx')
# %%
