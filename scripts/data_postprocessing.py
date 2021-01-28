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
                network = pypsa.Network('inter_results/'+file)
                sol.put(network)
                del(network)


#%%
if __name__=='__main__':
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        try:
            snakemake = mock_snakemake('data_postprocess')
            os.chdir('..')
        except :
            os.chdir('..')
            snakemake = mock_snakemake('data_postprocess')
            #os.chdir('..')
            
    configure_logging(snakemake,skip_handlers=True)


    dir_lst = os.listdir('inter_results/')
    network = pypsa.Network('inter_results/'+dir_lst[0])
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


    sol.save_csv('results/result_')
    #sol.save_xlsx('results/result.xlsx')
# %%
