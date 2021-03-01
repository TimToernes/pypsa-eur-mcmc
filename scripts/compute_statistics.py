"""
The script will compute relevant statistics for the mcmc chain
"""
#%%
import pandas as pd
import numpy as np
import os 
import seaborn as sns
import matplotlib.pyplot as plt
from _helpers import configure_logging
#%% Data import 


theta = pd.read_csv('inter_results/theta.csv',index_col=0)
means = theta.mean()[0:33]
stds = theta.std()[0:33]

theta.columns = [f'theta_{x}' for x in range(33)]+['s','c','a']
theta['id'] = theta.index

accept_percent = sum(theta.a)/theta.shape[0]*100
print(f'Acceptance {accept_percent:.1f}%')
print('Means:\n',means)

theta_long = pd.wide_to_long(theta,stubnames=['theta_'],i='id',j='theta')
theta_long = theta_long.reset_index(level=['theta'])

#%% Plot of chain development over time 
sns.set_theme(style="ticks")
# Define the palette as a list to specify exact values
palette = sns.color_palette("rocket", as_cmap=True)

# Plot the lines on two facets
sns.relplot(
    data=theta_long,
    x="s", y="theta_",
    hue="theta",
    palette=palette,
    col='c',
    kind="line",
    height=5, aspect=.75, 
)

#%% Correlogram

# library & dataset
import matplotlib.pyplot as plt
import seaborn as sns
#df = sns.load_dataset('iris')
df = theta.iloc[:,10:20]
df['c'] = theta.c
# with regression
#sns.pairplot(df, kind="reg")
#plt.show()
 
# without regression
sns.pairplot(df, kind="hist", hue="c", diag_kind='kde')
plt.show()


#%%

def calc_gelman_rubenstain(theta):
    """
    Calculate the Gellman Rubenstain parameters for a chain.
    The esitmated parameter should be below 1.2 for all parameters to ensure convergence 
    https://blog.stata.com/2016/05/26/gelman-rubin-convergence-diagnostic-using-multiple-chains/
    """

    # Sample mean 
    theta_hat = theta.mean()[0:33]
    # Sample mean for the individual chains 
    theta_hat_m = theta.groupby('c').mean().values[:,0:33]
    # Variance for the individual chains
    sigma_hat_m = theta.groupby('c').var().values[:,0:33]
    # Number of samples in each chain 
    N = theta.shape[0]
    # Number of chains 
    M = len(theta.c.unique())
    # degrees of freedom 
    d = 33
    # Between chains variance 
    B = N/(M-1) * sum([(theta-theta_hat)**2 for theta in theta_hat_m])
    # Within chain variance 
    W = np.mean(sigma_hat_m,axis=0)
    # Pooled variance 
    V_hat = (N-1)/N * W + (M+1)/(M*N) * B
    # Potential Scale Reduction Factor (PSRF)
    psrf = np.sqrt((d+3)/(d+1)*V_hat/W)
                        
    return psrf

calc_gelman_rubenstain(theta)

#%%
if __name__=='__main__':
    if 'snakemake' not in globals():
        from _helpers import mock_snakemake
        try:
            snakemake = mock_snakemake('run_single_chain',chain=1,sample=101)
            os.chdir('..')
        except :
            
            os.chdir('..')
            snakemake = mock_snakemake('run_single_chain',chain=1,sample=101)

    configure_logging(snakemake,skip_handlers=True)

# %%
