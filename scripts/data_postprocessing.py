#%%
import pypsa
import os 
import csv
import numpy as np 
import re 
from _helpers import configure_logging
from solutions import solutions
import multiprocessing as mp
import queue # imported for using queue.Empty exception
from itertools import product
from functools import partial
import time 
from shutil import copyfile


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
            if file[:7] == 'network':
                network = pypsa.Network(f'inter_results/{snakemake.config["run_name"]}/'+file,override_component_attrs=override_component_attrs)
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


    dir_lst = os.listdir(f'inter_results/{snakemake.config["run_name"]}/')
    network = pypsa.Network(snakemake.input[0],override_component_attrs=override_component_attrs)
    man = mp.Manager()
    sol = solutions(network, man)
    q_in = man.Queue()
    q_proc_done = man.Queue()
        
    for file in dir_lst:
        q_in.put(file)

    processes = []
    for i in range(snakemake.threads):
        p = mp.Process(target=worker,args=(q_in,sol,q_proc_done))
        p.start()
        processes.append(p)
    
    while q_proc_done.qsize()<snakemake.threads:
        time.sleep(1)

    for p in processes:
        p.kill()
        p.join()

    sol.merge()


    sol.save_csv(f'results/{snakemake.config["run_name"]}/result_')

    #copyfile(f'inter_results/{snakemake.config["run_name"]}/theta.csv',f'results/{snakemake.config["run_name"]}/theta.csv')
    
    #sol.save_xlsx('results/result.xlsx')
# %%
