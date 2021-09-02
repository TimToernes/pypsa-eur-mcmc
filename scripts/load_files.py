# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import netCDF4 as nc
import os
import pypsa
import dask 
import dask.dataframe as dd
import pandas as pd
import warnings
import logging
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
logging.getLogger('pypsa').setLevel(logging.ERROR)
import time 



if __name__ == '__main__':
    from dask.distributed import Client
    #client = Client()
    # %%



    data_dir = '../inter_results/sweep_e_2030/'
    #data_dir = '../inter_results/test/'
    dir_list = os.listdir(data_dir)


    # %%
    n = len(dir_list)
    for i,f in enumerate(dir_list[::-1]):
        if f[-3:] != '.nc':
            dir_list.pop(n-i-1)


    # %%
    print(len(dir_list))


    # %%
    t = time.time()
    networks_d = [dask.delayed(pypsa.Network)(data_dir+f) for f in dir_list]
    dfs = [n.generators[['p_nom']].T for n in networks_d]
    dd_df = dd.from_delayed(dfs)
    dd_df.compute(num_workers=8)
    print('dask time elapsed: ',time.time()-t)


    # %%
    #dd_df.visualize()



    # %%
    t = time.time()
    networks = [pypsa.Network(data_dir+f) for f in dir_list]
    dfs = [n.generators.p_nom_opt for n in networks]
    df = pd.DataFrame(dfs)
    print('pandas time elapsed: ',time.time()-t)



