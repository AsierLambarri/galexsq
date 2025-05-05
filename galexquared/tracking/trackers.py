import os
import yt
import sys
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from pytreegrav import Potential, PotentialTarget

from .base import BaseTracker
from ..utils import particle_unbinding_fire

import multiprocessing as mp
from multiprocessing.dummy import Pool as ThreadPool
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import gc
from time import time


def _extract_particles_infall(subtree, sp, row, unyt_arr, ptype):
    """Returns particle indices that are bound (AGORA VII)
    """
    st = time()
    
    pos = sp[ptype, "particle_position"].to("kpc")
    mass = sp[ptype, "particle_mass"].to("Msun")
    
    mask = np.linalg.norm(pos - sp.center.to("kpc"), axis=1) < 0.5 * sp.radius.to("kpc")
    pos_src = pos[mask]
    mass_src = mass[mask]
    
    n = len(mass)
    m = len(mass_src)
    
    ft = time()
    potential = unyt_arr( PotentialTarget(
        pos_target=pos, 
        softening_target=None, 

        pos_source=pos_src, 
        m_source=mass_src,
        softening_source=None, 
    
        G=4.300917270038E-6, 
    
        theta=min( 0.7, 0.4 * (n / 1E3) ** 0.1 ),
        parallel=True
    ),
    'km**2/s**2'
    )
    
    # potential = unyt_arr(Potential(
    #     pos=pos, 
    #     m=mass,
    #     softening=None, 
    #     G=4.300917270038E-6, 
    #     theta=min( 0.7, 0.4 * (n / 1E3) ** 0.1 ),
    #     parallel=True
    # ), 
    # 'km**2/s**2'
    # )
    
    ft1 = time()
    
    grav_energy = potential * sp[ptype, "particle_mass"].to("Msun")
    del potential, pos_src, mass_src, mask
    
    vel = unyt_arr(row[["velocity_x", "velocity_y", "velocity_z"]].values[0], 'km/s')
    kinetic = 0.5 * sp[ptype, "particle_mass"].to("Msun") * np.linalg.norm(sp[ptype, "particle_velocity"].to("km/s") - vel, axis=1)**2
    index_sel1 = sp[ptype, "particle_index"][kinetic + grav_energy < 0].astype(int).value
    
    del sp
    del grav_energy, kinetic
    gc.collect()
    ft2 = time()

    stdout = f"{subtree}, data extraction: {ft-st}s, Potential for n={n}/src={m}: {ft1 - ft}s, rest={ft2-ft1}s"
    return index_sel1, stdout


def _extract_particles_ptoinfall(subtree, file, row, ptype):
    """Returns particle indices that are bound (AGORA VII)
    """
    cen = row[["position_x", "position_y", "position_z"]].values[0]
    rvir = row["virial_radius"].values[0]

    ds = yt.load(file)
    sp = ds.sphere((cen, 'kpccm'), (rvir, 'kpccm'))
    
    return sp[ptype, "particle_index"].astype(int).value


def _extract_particles_infall_wrapper(task):
    return _extract_particles_infall(*task)
def _extract_particles_ptoinfall_wrapper(task):
    return _extract_particles_ptoinfall(*task)




class GravityTracker(BaseTracker):
    """Tracker that uses pytreegrav.
    """
    def __init__(self, 
             sim_dir,
             catalogue,
             equivalence_table,
             ptypes
             ):
        """Init function.
        """
        super().__init__(sim_dir, catalogue, equivalence_table, ptypes)



    def _select_particles(self, inserted_halos, **kwargs):
        """Selects particles to track for a inserted subtree's.
        """
        yt.utilities.logger.ytLogger.setLevel(40)

        nproc = min(int(kwargs.get("parallel", 1)), len(inserted_halos))
        pool = mp.Pool(nproc)

        if nproc > 6: warnings.warn(f"Having nproc={nproc} may fuck up your cache.")

        unyt_arr = self.ds.arr

        tasks_infall = []
        tasks_prior = []
        for subtree in inserted_halos:
            merge = self.MergeInfo[self.MergeInfo["Sub_tree_id"] == subtree]
            
            row_infall = self.mergertree.subtree(subtree, snapshot=merge["crossing_snap"].values[0])
            row_prior = self.mergertree.subtree(subtree, snapshot=merge["crossing_snap_2"].values[0])

            assert int(self.equiv[self.equiv["snapshot"] == merge["crossing_snap"].values[0]].index[0]) == self._snap_to_index(merge["crossing_snap"].values[0]), "file selection fucked up"
            
            file_prior = self._files[self._snap_to_index(merge["crossing_snap_2"].values[0])]

            
            cen = row_infall[["position_x", "position_y", "position_z"]].values[0]
            rvir = row_infall["virial_radius"].values[0]
            
            sp = self.ds.sphere((cen, 'kpccm'), (2 * rvir, 'kpccm'))
            
            tasks_infall.append((
                subtree,
                sp,
                row_infall,
                unyt_arr,
                self.ptypes["nbody"]
            ))
            
            tasks_prior.append((
                subtree,
                file_prior,
                row_prior,
                self.ptypes["nbody"]
            ))

            del sp
            


        chsize = kwargs.get("chunksize", len(tasks_prior) // (nproc) + 1)

        for i, particle_list in enumerate(
            tqdm(pool.imap(_extract_particles_ptoinfall_wrapper, tasks_prior, chunksize=chsize), total=len(tasks_prior), desc="Selecting particles prior to infall...")
        ):
            self.particles_index_dict_prior[tasks_prior[i][0]] = particle_list
  
        pool.close()
        pool.join()
        del pool
        
        stdout = []
        for i, task in enumerate(
            tqdm(tasks_infall, total=len(tasks_infall), desc="Selecting particles at infall...")
        ):
            self.particles_index_dict_infall[task[0]], out = _extract_particles_infall_wrapper(task) 
            stdout.append(out)
            

        print('\n'.join(stdout))
        yt.utilities.logger.ytLogger.setLevel(20)

        del tasks_prior, tasks_infall, particle_list 



    def track_halos(self, halos_tbt, output, **kwargs):
        """Tracks the halos_tbt (to-be-tracked, not throw-back-thursday) halos. Each halos starting poit can be a different snapshot
        """
        if isinstance(halos_tbt, pd.DataFrame):
            self.halos_tbt = halos_tbt.sort_values("Snapshot")
        else: 
            self.halos_tbt = pd.read_csv(halos_tbt).sort_values("Snapshot")

        self._postprocess_trees(self.halos_tbt["Sub_tree_id"].unique())
        
        self.output = output
        try:
            self.close()
        except:
            pass
            
        if not isinstance(self.CompleteTree, pd.DataFrame) or not isinstance(self.equiv, pd.DataFrame):
            raise ValueError("Either CompleteTree or equivalence_table are not pandas dataframes! or are not properly set!")
        
        self._sanitize_merge_info()
        self._create_snapztdir()
        self._create_thtrees()
        
        start_snapst = dict( zip(self.MergeInfo["Sub_tree_id"].values,  self.MergeInfo["Snapshot"].values) )
        file_path = Path(output)
        if file_path.is_file():
            outpùt2 = output[:-5] + "_v2" + output[-5:]
            warnings.warn(f"The file {output} already exists", RuntimeWarning)
            
        self._create_hdf5_file(fname=output)

        active_halos = set()
        terminated_halos = set()
        live_halos = set()

        ref_table = self.MergeInfo
        self.particles_index_dict = {}
        
        for index, row in self.snap_z_t_dir.iterrows():
            snapshot, redshift, time, pdir = row["Snapshot"], row["Redshift"], row["Time"], row["pdir"]
            
            inserted_halos = ref_table[ref_table["Snapshot"] == snapshot]["Sub_tree_id"].astype(int).values
            active_halos.update(inserted_halos)
                
            live_halos = active_halos - terminated_halos
            if not live_halos:
                continue
            if not active_halos:
                raise Exception(f"You dont have any active halos at this point! (snapshot {snapshot}). This means that something went wrong!")
            else:
                self.ds = yt.load(pdir)

            self._select_particles(inserted_halos, **kwargs)


            del self.ds

        self.f.close()
        return active_halos, live_halos, terminated_halos


        def close(self):
            os.remove(output)
            self.f.close()




        
            
               # for subtree in live_halos:
               # subtree_table = self.halos_tbt[self.halos_tbt["Sub_tree_id"] == subtree]
               # if subtree in inserted_halos:
               #     if subtree_table["R/Rvir"] < 1:
               #         insert_row = 1
               #     else:
               #         insert_row = subtree_table
               #         
               #     sp = self.ds.sphere(
               #         self.ds.arr(subtree_table[["position_x", "position_y", "position_z"]].astype(float).values, 'kpccm'),    
               #         self.ds.quan(subtree_table["virial_radius"].astype(float).values, 'kpccm')
               #     )
               #     particles = HaloParticlesOutside(subtree, sp)
               #     
               # else:
               #     pass
                #pot + trackeo
                #add particles to self.f
                #compute statistics
                #add statistics to csv
#            for subtree in live_halos:
                #check for merging with main halo or merges between them using csvs. Now all of them are up to date.     







































    
    