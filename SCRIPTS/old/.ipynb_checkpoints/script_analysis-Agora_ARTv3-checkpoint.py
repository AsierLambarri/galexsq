import yt
import numpy as np
import pandas as pd
from astropy.table import Table
from unyt import unyt_quantity, unyt_array
from yt.utilities.logger import ytLogger
from uncertainties import ufloat

import concurrent.futures

import galexquared as gal
from galexquared.class_methods import random_vector_spherical

#ytLogger.setLevel(50)





gal.config.code = "ART"
#files = ["ART_satellitesV5_Rvir1.0.csv", "ART_satellitesV5_Rvir1.5.csv", "ART_satellitesV5_Rvir2.0.csv", "ART_satellitesV5_Rvir3.0.csv", "ART_satellitesV5_Rvir4.0.csv", "ART_satellitesV5_Rvir5.0.csv"]
files = ["ART_satellitesV5_Rvir1.5.csv"]

for f in files:
    print(f"ANALYZING FILE {f}")
    tdir = "/home/asier/StructuralAnalysis/satellite_tables/"
    errors = []
    
    file = tdir + f
    
    candidates = pd.read_csv(file)
    
    candidates['delta_rel'] = pd.Series()
    candidates['stellar_mass'] = pd.Series()
    candidates['dark_mass'] = pd.Series()
    candidates['all_gas_mass'] = pd.Series()
    
    candidates['rh_stars_physical'] = pd.Series()
    candidates['rh_dm_physical'] = pd.Series()
    candidates['e_rh_stars_physical'] = pd.Series()
    candidates['e_rh_dm_physical'] = pd.Series()
    
    candidates['rh3D_stars_physical'] = pd.Series()
    candidates['rh3D_dm_physical'] = pd.Series()

    candidates['Mhl'] = pd.Series()
    
    candidates['sigma*'] = pd.Series()
    candidates['e_sigma*'] = pd.Series()
    
    candidates['Mdyn'] = pd.Series()
    candidates['e_Mdyn'] = pd.Series()

    MT = gal.MergerTree("../merger_trees_csv/ART_CompleteTree.csv")
    MT.set_equivalence("ART_equivalence.dat")
    pdir = "/media/asier/EXTERNAL_USBA/Cosmo_v18/"
    
    for index, halorow in candidates.iterrows():
        print(f"#####   {halorow['Sub_tree_id']}:snapshot-{halorow['Snapshot']}   #####\n")
            
        lines_of_sight = random_vector_spherical(N=20)

        try:
            halo = MT.load_halo(pdir, sub_tree=halorow["Sub_tree_id"], snapshot=halorow["Snapshot"])        
        
            if halo.stars.empty:
                errors.append(f"Sub_tree_id {halorow['Sub_tree_id']} in snapshot-{halorow['Snapshot']} has no stars at all.")
                continue
                
            halo.compute_energy(refine=False)
            halo.compute_stars_in_halo()
            halo.switch_to_bound()
            
            if halo.stars["mass"].sum() == 0:
                errors.append(f"Sub_tree_id {halorow['Sub_tree_id']} in snapshot-{halorow['Snapshot']} has no stars bound.")
                continue
            
            halo.stars.refined_center6d(nmin=40)
            halo.darkmatter.refined_center6d(nmin=50)
    
            candidates.loc[index, "all_gas_mass"] =  halo.gas["mass"].sum().to("Msun").value
            candidates.loc[index, "dark_mass"] =  halo.darkmatter["mass"].sum().to("Msun").value
            candidates.loc[index, "stellar_mass"] =  halo.stars["mass"].sum().to("Msun").value
            candidates.loc[index, "delta_rel"] = float(halo.stars.delta_rel)
    
            candidates.loc[index, 'rh3D_stars_physical'] = halo.stars.half_mass_radius()[0].in_units("kpc").value
            candidates.loc[index, 'rh3D_dm_physical'] = halo.darkmatter.half_mass_radius()[0].in_units("kpc").value
    
            rh_stars = halo.stars.half_mass_radius(project=True, lines_of_sight=lines_of_sight).to("kpc").value
            rh_dm = halo.darkmatter.half_mass_radius(project=True, lines_of_sight=lines_of_sight).to("kpc").value
            sigma = halo.stars.los_dispersion(lines_of_sight=lines_of_sight).to("km/s").value
            mdyn = 580 * 1E3 * rh_stars * sigma ** 2
            
            candidates.loc[index, "rh_stars_physical"] = rh_stars.mean()
            candidates.loc[index, "e_rh_stars_physical"] =  rh_stars.std()
            candidates.loc[index, "rh_dm_physical"] =  rh_dm.mean()
            candidates.loc[index, "e_rh_dm_physical"] = rh_dm.std()
            candidates.loc[index, "sigma*"] = sigma.mean()
            candidates.loc[index, "e_sigma*"] = sigma.std()
            candidates.loc[index, 'Mhl'] = halo.enclosed_mass(halo.stars.q["cm"], halo.stars.q["rh3d"], components="all").to("Msun").value
            
            candidates.loc[index, 'Mdyn'] = mdyn.mean()
            candidates.loc[index, 'e_Mdyn'] = mdyn.std()
        
        
        except:
            errors.append(f"Strange error ocurred in {halorow['Sub_tree_id']} in snapshot-{halorow['Snapshot']}")
            continue
        
    candidates.to_csv(file, index=False)



print(f"Finished")










