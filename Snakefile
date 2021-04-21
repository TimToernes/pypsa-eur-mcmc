
#configfile: "config.yaml"

wildcard_constraints:
    sample="[1-9]?[0-9]?[0-9]?[0-9]"

chains = range(config['sampler']['chains'])

rule all:
    input:
        expand("inter_results/{run_name}/network_c{chain}_s{sample}.nc",chain=chains,sample=config['sampler']['samples'],run_name=config['run_name']),
        "inter_results/{run_name}/sigma_s{sample}.csv".format(sample=config['sampler']['samples'],run_name=config['run_name'])



rule initialize_networks:
    input:
        network=config['network']
    output:
        #variables_file = 'results/variables.csv'
        expand("inter_results/{run_name}/network_c{chain}_s1.nc",chain=chains,run_name = config['run_name']),
        f"inter_results/{config['run_name']}/sigma_s1.csv",
        f"results/{config['run_name']}/config.yaml",
        
    threads: config['solving']['solver']['threads']
    script: 'scripts/initialize_network.py'
    
def chain_input(w):
    out = []
    out.append('inter_results/{run_name}/network_c{chain}_s{sample}.nc'.format(chain=w.chain,sample=int(w.sample)-config['sampler']['batch'],run_name=w.run_name))
    out.append('inter_results/{run_name}/sigma_s{sample}.csv'.format(sample=int(w.sample)-config['sampler']['batch'],run_name=w.run_name))
    return out 

rule run_single_chain:
    input:
        chain_input
        #network = lambda w: 'inter_results/network_c{chain}_s{sample}.nc'.format(chain=w.chain,sample=int(w.sample)-100)
        #sigma = lambda w: 'inter_results/sigma_s{sample}.csv'.format(sample=int(w.sample)-100)
    output:
        "inter_results/{run_name}/network_c{chain}_s{sample}.nc"
    threads: 4
    script: 'scripts/mh_chain.py'


def sigma_input(w):
    out = []
    out.append('inter_results/{run_name}/sigma_s{sample}.csv'.format(sample=int(w.sample)-config['sampler']['batch'],run_name=w.run_name))
    #out.append('inter_results/sigma_s{sample}.csv'.format(sample=int(w.sample)-100))
    for c in range(config['sampler']['chains']):
        out.append('inter_results/{run_name}/network_c{chain}_s{sample}.nc'.format(chain=c,sample=int(w.sample),run_name=w.run_name))
    
    return out 

rule calc_sigma:
    input:
        sigma_input,
        #sigma = 'inter_results/{run_name}/sigma_s{sample_m1}.csv'.format(sample_m1=sample-snakemake.config['sampler']['batch'])
    output:
        sigma = 'inter_results/{run_name}/sigma_s{sample}.csv',
        #theta = 'inter_results/theta_s{sample}.csv'
    threads: 4
    script: 
        'scripts/calc_sigma.py'


rule data_postprocess:
    input:
        expand("inter_results/{run_name}/network_c{chain}_s{sample}.nc",chain=chains,sample=config['sampler']['samples'],run_name=config['run_name']),
        sigma = 'inter_results/{run_name}/sigma_s{sample}.csv'.format(run_name=config['run_name'],sample=config['sampler']['samples'])
    threads: 64
    resources: mem='400G'
    script:
        'scripts/data_postprocessing.py'

rule speed_test:
    threads: 4
    script: 'scripts/solve_speed_test.py'

rule co2_sweep:
    input:
        network=config['network']
    threads: 4
    script: 'scripts/co2_sweep.py'
