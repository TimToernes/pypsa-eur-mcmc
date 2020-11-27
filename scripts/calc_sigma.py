#%%
import pypsa
import os 
import csv
import numpy as np 
import re 
from _helpers import configure_logging
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

def get_theta(netwrok,mcmc_variables):
    theta = []
    for var in mcmc_variables:
        if network.generators.index.isin(var).any():
            theta.append(network.generators.p_nom_opt.loc[network.generators.index.isin(var)].sum())
        else :
            theta.append(network.storage_units.p_nom_opt.loc[network.storage_units.index.isin(var)].sum())

    return theta


#%%
if __name__=='__main__':
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        try:
            snakemake = mock_snakemake('calc_sigma',sample=201)
        except :
            os.chdir(os.getcwd()+'/scripts')
            snakemake = mock_snakemake('calc_sigma',sample=101)
            os.chdir('..')
            
    configure_logging(snakemake,skip_handlers=True)

    # impor a network 
    network = pypsa.Network(snakemake.input[0])
    mcmc_variables = read_csv(network.mcmc_variables)
    eps = snakemake.config['sampler']['eps']


    thetas = []
    # Itterate over all files in the inter_results directory 
    # If the file is a network file, it is loaded and the theta value is extracted and stored to the thetas list
    dir_lst = os.listdir('inter_results/')
    for file in dir_lst:
        if file[:7] == 'network':
            network = pypsa.Network('inter_results/'+file)
            theta = get_theta(network,mcmc_variables)
            thetas.append(theta)
            del(network)
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
