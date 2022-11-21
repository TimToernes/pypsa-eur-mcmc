## 30.000 ways to reach 55% decarbonization of the European electricity sector

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository contains all files nececary for replicating the results in the resarch article "30.000 ways to reach 55% decarbonization of the European electricity sector". 

To reproduce results clone this reporsitory, install the anaconda environment provided with the environment.yaml file, and run the Snakemake workflow with the following command:

<code> snakemake data_postprocess --configfile config_scenarios.yaml --nolock </code>

It is recomended that the simulations are run on a HPC.