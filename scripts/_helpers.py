# SPDX-FileCopyrightText: : 2017-2020 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pandas as pd
from pathlib import Path



def configure_logging(snakemake, skip_handlers=False):
    """
    Configure the basic behaviour for the logging module.

    Note: Must only be called once from the __main__ section of a script.

    The setup includes printing log messages to STDERR and to a log file defined
    by either (in priority order): snakemake.log.python, snakemake.log[0] or "logs/{rulename}.log".
    Additional keywords from logging.basicConfig are accepted via the snakemake configuration
    file under snakemake.config.logging.

    Parameters
    ----------
    snakemake : snakemake object
        Your snakemake object containing a snakemake.config and snakemake.log.
    skip_handlers : True | False (default)
        Do (not) skip the default handlers created for redirecting output to STDERR and file.
    """

    import logging

    kwargs = snakemake.config.get('logging', dict())
    kwargs.setdefault("level", "INFO")

    if skip_handlers is False:
        fallback_path = Path(__file__).parent.joinpath('..', 'logs', f"{snakemake.rule}.log")
        logfile = snakemake.log.get('python', snakemake.log[0] if snakemake.log
                                    else fallback_path)
        kwargs.update(
            {'handlers': [
                # Prefer the 'python' log, otherwise take the first log for each
                # Snakemake rule
                logging.FileHandler(logfile),
                logging.StreamHandler()
                ]
            })
    logging.basicConfig(**kwargs)

def load_network(import_name=None, custom_components=None):
    """
    Helper for importing a pypsa.Network with additional custom components.

    Parameters
    ----------
    import_name : str
        As in pypsa.Network(import_name)
    custom_components : dict
        Dictionary listing custom components.
        For using ``snakemake.config['override_components']``
        in ``config.yaml`` define:

        .. code:: yaml

            override_components:
                ShadowPrice:
                    component: ["shadow_prices","Shadow price for a global constraint.",np.nan]
                    attributes:
                    name: ["string","n/a","n/a","Unique name","Input (required)"]
                    value: ["float","n/a",0.,"shadow value","Output"]

    Returns
    -------
    pypsa.Network
    """

    import pypsa
    from pypsa.descriptors import Dict

    override_components = None
    override_component_attrs = None

    if custom_components is not None:
        override_components = pypsa.components.components.copy()
        override_component_attrs = Dict({k : v.copy() for k,v in pypsa.components.component_attrs.items()})
        for k, v in custom_components.items():
            override_components.loc[k] = v['component']
            override_component_attrs[k] = pd.DataFrame(columns = ["type","unit","default","description","status"])
            for attr, val in v['attributes'].items():
                override_component_attrs[k].loc[attr] = val

    return pypsa.Network(import_name=import_name,
                         override_components=override_components,
                         override_component_attrs=override_component_attrs)

def pdbcast(v, h):
    return pd.DataFrame(v.values.reshape((-1, 1)) * h.values,
                        index=v.index, columns=h.index)

def load_network_for_plots(fn, tech_costs, config, combine_hydro_ps=True):
    import pypsa
    from add_electricity import update_transmission_costs, load_costs

    network = pypsa.Network(fn)

    network.loads["carrier"] = network.loads.bus.map(network.buses.carrier) + " load"
    network.stores["carrier"] = network.stores.bus.map(network.buses.carrier)

    network.links["carrier"] = (network.links.bus0.map(network.buses.carrier) + "-" + network.links.bus1.map(network.buses.carrier))
    network.lines["carrier"] = "AC line"
    network.transformers["carrier"] = "AC transformer"

    network.lines['s_nom'] = network.lines['s_nom_min']
    network.links['p_nom'] = network.links['p_nom_min']

    if combine_hydro_ps:
        network.storage_units.loc[network.storage_units.carrier.isin({'PHS', 'hydro'}), 'carrier'] = 'hydro+PHS'

    # #if the carrier was not set on the heat storage units
    # bus_carrier = n.storage_units.bus.map(n.buses.carrier)
    # n.storage_units.loc[bus_carrier == "heat","carrier"] = "water tanks"

    Nyears = network.snapshot_weightings.sum()/8760.
    costs = load_costs(Nyears, tech_costs, config['costs'], config['electricity'])
    update_transmission_costs(network, costs)

    return network

def aggregate_p_nom(n):
    return pd.concat([
        n.generators.groupby("carrier").p_nom_opt.sum(),
        n.storage_units.groupby("carrier").p_nom_opt.sum(),
        n.links.groupby("carrier").p_nom_opt.sum(),
        n.loads_t.p.groupby(n.loads.carrier,axis=1).sum().mean()
    ])

def aggregate_p(n):
    return pd.concat([
        n.generators_t.p.sum().groupby(n.generators.carrier).sum(),
        n.storage_units_t.p.sum().groupby(n.storage_units.carrier).sum(),
        n.stores_t.p.sum().groupby(n.stores.carrier).sum(),
        -n.loads_t.p.sum().groupby(n.loads.carrier).sum()
    ])

def aggregate_e_nom(n):
    return pd.concat([
        (n.storage_units["p_nom_opt"]*n.storage_units["max_hours"]).groupby(n.storage_units["carrier"]).sum(),
        n.stores["e_nom_opt"].groupby(n.stores.carrier).sum()
    ])

def aggregate_p_curtailed(n):
    return pd.concat([
        ((n.generators_t.p_max_pu.sum().multiply(n.generators.p_nom_opt) - n.generators_t.p.sum())
         .groupby(n.generators.carrier).sum()),
        ((n.storage_units_t.inflow.sum() - n.storage_units_t.p.sum())
         .groupby(n.storage_units.carrier).sum())
    ])

def aggregate_costs(n, flatten=False, opts=None, existing_only=False):
    from six import iterkeys, itervalues

    components = dict(Link=("p_nom", "p0"),
                      Generator=("p_nom", "p"),
                      StorageUnit=("p_nom", "p"),
                      Store=("e_nom", "p"),
                      Line=("s_nom", None),
                      Transformer=("s_nom", None))

    costs = {}
    for c, (p_nom, p_attr) in zip(
        n.iterate_components(iterkeys(components), skip_empty=False),
        itervalues(components)
    ):
        if not existing_only: p_nom += "_opt"
        costs[(c.list_name, 'capital')] = (c.df[p_nom] * c.df.capital_cost).groupby(c.df.carrier).sum()
        if p_attr is not None:
            p = c.pnl[p_attr].sum()
            if c.name == 'StorageUnit':
                p = p.loc[p > 0]
            costs[(c.list_name, 'marginal')] = (p*c.df.marginal_cost).groupby(c.df.carrier).sum()
    costs = pd.concat(costs)

    if flatten:
        assert opts is not None
        conv_techs = opts['conv_techs']

        costs = costs.reset_index(level=0, drop=True)
        costs = costs['capital'].add(
            costs['marginal'].rename({t: t + ' marginal' for t in conv_techs}),
            fill_value=0.
        )

    return costs

def progress_retrieve(url, file):
    import urllib
    from progressbar import ProgressBar

    pbar = ProgressBar(0, 100)

    def dlProgress(count, blockSize, totalSize):
        pbar.update( int(count * blockSize * 100 / totalSize) )

    urllib.request.urlretrieve(url, file, reporthook=dlProgress)


def mock_snakemake(rulename, **wildcards):
    """
    This function is expected to be executed from the 'scripts'-directory of '
    the snakemake project. It returns a snakemake.script.Snakemake object,
    based on the Snakefile.

    If a rule has wildcards, you have to specify them in **wildcards.

    Parameters
    ----------
    rulename: str
        name of the rule for which the snakemake object should be generated
    **wildcards:
        keyword arguments fixing the wildcards. Only necessary if wildcards are
        needed.
    """
    import snakemake as sm
    import os
    from pypsa.descriptors import Dict
    from snakemake.script import Snakemake

    script_dir = Path(__file__).parent.resolve()
    #assert Path.cwd().resolve() == script_dir, \
    #  f'mock_snakemake has to be run from the repository scripts directory {script_dir}'
    #os.chdir(script_dir.parent)
    for p in sm.SNAKEFILE_CHOICES:
        if os.path.exists(p):
            snakefile = p
            break
    workflow = sm.Workflow(snakefile)
    workflow.include(snakefile)
    workflow.global_resources = {}
    rule = workflow.get_rule(rulename)
    dag = sm.dag.DAG(workflow, rules=[rule])
    wc = Dict(wildcards)
    job = sm.jobs.Job(rule, dag, wc)

    def make_accessable(*ios):
        for io in ios:
            for i in range(len(io)):
                io[i] = os.path.abspath(io[i])

    make_accessable(job.input, job.output, job.log)
    snakemake = Snakemake(job.input, job.output, job.params, job.wildcards,
                          job.threads, job.resources, job.log,
                          job.dag.workflow.config, job.rule.name, None,)
    # create log and output dir if not existent
    for path in list(snakemake.log) + list(snakemake.output):
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    #os.chdir(script_dir)
    return snakemake


def make_override_component_attrs():
    import pypsa
    import numpy as np
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
    override_component_attrs["StorageUnit"].loc["p_dispatch"] = ["series","MW",0.,"Storage discharging.","Output"]
    override_component_attrs["StorageUnit"].loc["p_store"] = ["series","MW",0.,"Storage charging.","Output"]
    return override_component_attrs



def get_tech_colors():
    tech_colors = {
    # wind,
    'onwind': "#235ebc",
    'onshore wind': "#235ebc",
    'wind': "#235ebc",
    'offwind': "#6895dd",
    'offshore wind': "#6895dd",
    'offwind-ac': "#6895dd",
    'offshore wind (AC)': "#6895dd",
    'offwind-dc': "#74c6f2",
    'offshore wind (DC)': "#74c6f2",
    # water,
    'hydro': '#298c81',
    'hydro reservoir': '#298c81',
    'ror': '#3dbfb0',
    'run of river': '#3dbfb0',
    'hydroelectricity': '#298c81',
    'PHS': '#51dbcc',
    'wave': '#a7d4cf',
    # solar,
    'solar': "#f9d002",
    'solar PV': "#f9d002",
    'solar thermal': '#ffbf2b',
    'solar rooftop': '#ffea80',
    # gas,
    'OCGT': '#e0986c',
    'OCGT marginal': '#e0986c',
    'OCGT-heat': '#e0986c',
    'gas boiler': '#db6a25',
    'gas boilers': '#db6a25',
    'gas boiler marginal': '#db6a25',
    'gas': '#FFA500', #'#e05b09',
    'fossil gas': '#FFA500', #'#e05b09',
    'natural gas': '#FFA500', #'#e05b09',
    'CCGT': '#a85522',
    'CCGT marginal': '#a85522',
    'gas for industry co2 to atmosphere': '#692e0a',
    'gas for industry co2 to stored': '#8a3400',
    'gas for industry': '#853403',
    'gas for industry CC': '#692e0a',
    'gas pipeline': '#ebbca0',
    'gas pipeline new': '#a87c62',
    'LNG': '#0a335e',
    # oil,
    'oil': '#c9c9c9',
    'oil boiler': '#adadad',
    'agriculture machinery oil': '#949494',
    'shipping oil': "#808080",
    'land transport oil': '#afafaf',
    # nuclear,
    'Nuclear': '#FC6400', #'#ff8c00',
    'Nuclear marginal': '#FC6400', #'#ff8c00',
    'nuclear': '#FC6400', #'#ff8c00',
    'uranium': '#FC6400', #'#ff8c00',
    # coal,
    'Coal': '#545454',
    'coal': '#545454',
    'coal CC': '#545454',
    'Coal marginal': '#545454',
    'solid': '#545454',
    'Lignite': '#826837',
    'lignite': '#826837',
    'Lignite marginal': '#826837',
    # biomass,
    'biogas': '#e3d37d',
    'biomass': '#baa741',
    'biomass CHP': '#baa741',
    'other renewables and biofuels': '#baf238',
    'residential rural biomass boiler': '#baf238',
    'biomass boiler': '#baf238',
    'solid biomass': '#baa741',
    'solid biomass transport': '#baa741',
    'solid biomass for industry': '#7a6d26',
    'solid biomass for industry CC': '#47411c',
    'solid biomass for industry co2 from atmosphere': '#736412',
    'solid biomass for industry co2 to stored': '#47411c',
    # power transmission,
    'lines': '#6c9459',
    'transmission lines': '#fc03db',
    'electricity distribution grid': '#97ad8c',
    # electricity demand,
    'Electric load': '#110d63',
    'electric demand': '#110d63',
    'electricity': '#110d63',
    'industry electricity': '#2d2a66',
    'industry new electricity': '#2d2a66',
    'agriculture electricity': '#494778',
    # battery + EVs,
    'battery': '#ace37f',
    'battery storage': '#ace37f',
    'home battery': '#80c944',
    'home battery storage': '#80c944',
    'BEV charger': '#baf238',
    'battery charger':'#baf238',
    'battery discharger':'#e5ffa8',
    'V2G': '#e5ffa8',
    'land transport EV': '#baf238',
    'Li ion': '#baf238',
    # hot water storage,
    'water tanks': '#e69487',
    'hot water storage': '#e69487',
    'hot water charging': '#e69487',
    'hot water discharging': '#e69487',
    # heat demand,
    'Heat load': '#cc1f1f',
    'heat': '#cc1f1f',
    'heat demand': '#cc1f1f',
    'rural heat': '#ff5c5c',
    'central heat': '#cc1f1f',
    'decentral heat': '#750606',
    'low-temperature heat for industry': '#8f2727',
    'process heat': '#ff0000',
    'agriculture heat': '#d9a5a5',
    # heat supply,
    'heat pumps': '#2fb537',
    'heat pump': '#2fb537',
    'air heat pump': '#36eb41',
    'ground heat pump': '#2fb537',
    'Ambient': '#98eb9d',
    'gas CHP': '#8a5751',
    'CHP': '#8a5751',
    'CHP CC': '#634643',
    'CHP heat': '#8a5751',
    'CHP electric': '#8a5751',
    'district heating': '#e8beac',
    'resistive heater': '#d8f9b8',
    'retrofitting': '#8487e8',
    'building retrofitting': '#8487e8',
    # hydrogen,
    'H2 for industry': "#f073da",
    'H2 for shipping': "#ebaee0",
    'H2': '#bf13a0',
    'hydrogen': '#bf13a0',
    'SMR': '#870c71',
    'SMR CC': '#4f1745',
    'H2 liquefaction': '#d647bd',
    'hydrogen storage': '#bf13a0',
    'H2 storage': '#bf13a0',
    'land transport fuel cell': '#6b3161',
    'H2 pipeline': '#f081dc',
    'H2 pipeline retrofitted': '#ba99b5',
    'H2 Fuel Cell': '#c251ae',
    'H2 Electrolysis': '#ff29d9',
    # syngas,
    'Sabatier': '#9850ad',
    'methanation': '#c44ce6',
    'methane': '#c44ce6',
    'helmeth': '#e899ff',
    # synfuels,
    'Fischer-Tropsch': '#25c49a',
    'liquid': '#25c49a',
    'kerosene for aviation': '#a1ffe6',
    'naphtha for industry': '#57ebc4',
    # co2,
    'CC': '#f29dae',
    'CCS': '#f29dae',
    'CO2 sequestration': '#f29dae',
    'DAC': '#ff5270',
    'co2 stored': '#f2385a',
    'co2': '#f29dae',
    'co2 vent': '#ffd4dc',
    'CO2 pipeline': '#f5627f',
    # emissions,
    'process emissions CC': '#000000',
    'process emissions': '#222222',
    'process emissions to stored': '#444444',
    'process emissions to atmosphere': '#888888',
    'oil emissions': '#aaaaaa',
    'shipping oil emissions': "#555555",
    'land transport oil emissions': '#777777',
    'agriculture machinery oil emissions': '#333333',
    # other,
    'shipping': '#03a2ff',
    'power-to-heat': '#2fb537',
    'power-to-gas': '#c44ce6',
    'power-to-H2': '#ff29d9',
    'power-to-liquid': '#25c49a',
    'gas-to-power/heat': '#ee8340',
    'waste': '#e3d37d',
    'other': '#000000',}
    return tech_colors