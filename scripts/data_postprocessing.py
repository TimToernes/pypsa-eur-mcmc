#%%
import pypsa
import os 
import csv
import numpy as np 
import re 
from _helpers import configure_logging
from solutions import solutions
import multiprocessing as mp
from itertools import product
from functools import partial

#%%
if __name__=='__main__':
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        try:
            snakemake = mock_snakemake('data_postprocess')
        except :
            os.chdir(os.getcwd()+'/scripts')
            snakemake = mock_snakemake('data_postprocess')
            os.chdir('..')
            
    configure_logging(snakemake,skip_handlers=True)


    dir_lst = os.listdir('inter_results/')
    network = pypsa.Network('inter_results/'+dir_lst[0])
    sol = solutions(network, mp.Manager())

        
    for file in dir_lst:
        if file[:7] == 'network':
            network = pypsa.Network('inter_results/'+file)

            sol.put(network)
            del(network)

    sol.merge()

    sol.save_xlsx('results/result.xlsx')
# %%
