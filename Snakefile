
configfile: "config.yaml"

wildcard_constraints:
    sample="[1-9]?[0-9]?[0-9]?[0-9]"

chains = range(config['sampler']['chains'])

rule all:
    input:
        expand("inter_results/network_c{chain}_s{sample}.nc",chain=chains,sample=config['sampler']['samples'])



rule initialize_networks:
    input:
        network=config['network']
    output:
        #variables_file = 'results/variables.csv'
        expand("inter_results/network_c{chain}_s1.nc",chain=chains),
        "inter_results/sigma_s1.csv"
        
    threads: config['solver']['solver_options']['threads']
    script: 'scripts/initialize_network.py'
    
def chain_input(w):
    out = []
    out.append('inter_results/network_c{chain}_s{sample}.nc'.format(chain=w.chain,sample=int(w.sample)-100))
    out.append('inter_results/sigma_s{sample}.csv'.format(sample=int(w.sample)-100))
    return out 

rule run_single_chain:
    input:
        chain_input
        #network = lambda w: 'inter_results/network_c{chain}_s{sample}.nc'.format(chain=w.chain,sample=int(w.sample)-100)
        #sigma = lambda w: 'inter_results/sigma_s{sample}.csv'.format(sample=int(w.sample)-100)
    output:
        "inter_results/network_c{chain}_s{sample}.nc"
    threads: 4
    script: 'scripts/mh_chain.py'


def sigma_input(w):
    out = []
    #out.append('inter_results/sigma_s{sample}.csv'.format(sample=int(w.sample)-100))
    for c in range(config['sampler']['chains']):
        out.append('inter_results/network_c{chain}_s{sample}.nc'.format(chain=c,sample=int(w.sample)))
    return out 

rule calc_sigma:
    input:
        sigma_input
        #sigma = lambda w: 'inter_results/sigma_s{sample}.csv'.format(sample=int(w.sample)-100)
    output:
        sigma = 'inter_results/sigma_s{sample}.csv'
    script: 
        'scripts/calc_sigma.py'


rule data_postprocess:
    input:
        expand("inter_results/network_c{chain}_s{sample}.nc",chain=chains,sample=config['sampler']['samples'])
    output:
        'results/result.xlsx'
    script:
        'scripts/data_postprocessing.py'