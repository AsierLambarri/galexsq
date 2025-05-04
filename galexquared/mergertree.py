import os
import ytree
import warnings
import numpy as np
import pandas as pd; pd.options.display.max_columns = 1000
from tqdm import tqdm
from unyt import unyt_array, unyt_quantity

from copy import deepcopy

from .data_container import DataContainer
from .class_methods import load_ftable
from .config import config


class MergerTree:
    """Easy manage of merger trees, with the specific purpose of extracting galaxies with desired parameters and qualities. Although
    it may do more things than just that depending on what the stage you catch me -the author-, in.

    For now, it only accepts consistent-trees output.
    """
    def __init__(self,
                 fn,
                 arbor=None,
                ):
        """Initialization function.
        """
        self.fn = fn
        if isinstance(self.fn, str):
            if self.fn.endswith(".csv"):
                self.set_trees(self.fn)
                self._tree_loaded = True
            else:
                self.arbor = self._load_ytree_tree() if arbor is None else arbor
                self.size = self.arbor.size    
                self.all_fields = self.arbor.field_list
        
        elif isinstance(self.fn, pd.DataFrame):
             self.set_trees(self.fn)
        else:
            raise Exception("Format not supported!")
            
        self.set_fields({
                'Halo_ID': 'halo_id',
                'Snapshot': 'Snap_idx',
                'Redshift': 'redshift',
                'Time': ('time', 'Gyr'),
                'uid': 'uid',
                'desc_uid': 'desc_uid',
                'mass': ('mass', 'Msun'),
                'num_prog': 'num_prog',
                'virial_radius': ('virial_radius', 'kpc'),
                'scale_radius': ('scale_radius', 'kpc'),
                'vrms': ('velocity_dispersion', 'km/s'),
                'vmax': ('vmax', 'km/s'),
                'position_x': ('position_x', 'kpc'),
                'position_y': ('position_y', 'kpc'),
                'position_z': ('position_z', 'kpc'),
                'velocity_x': ('velocity_x', 'km/s'),
                'velocity_y': ('velocity_y', 'km/s'),
                'velocity_z': ('velocity_z', 'km/s'),
                'A[x]': 'A[x]',
                'A[y]': 'A[y]',
                'A[z]': 'A[z]',
                'b_to_a': 'b_to_a',
                'c_to_a': 'c_to_a',
                'T_U': 'T_|U|',
                'Tidal_Force': 'Tidal_Force',
                'Tidal_ID': 'Tidal_ID'
        })
        self.min_halo_mass = 1E7
        self.main_Rvir = -1
        self._computing_forest = False


    
    @property
    def CompleteTree(self):
        return self._CompleteTree
    @property
    def MainTree(self):
        return self._MainTree
    @property
    def SatelliteTrees(self):
        return self._SatelliteTrees
    @property
    def PrincipalLeaf(self):
        return self._PrincipalLeaf
    @property
    def equivalence_table(self):
        return self._equiv
    @property
    def ds(self):
        return self._ds
    
    def __repr__(self):
        if hasattr(self, "_CompleteTree"): return repr(self._CompleteTree)
        else: return ""

    def _load_ytree_tree(self):
        """Deletes arbor in temporary_arbor, if existing, and creates a new one for use in current instance.
        """
        sfn = "/".join(self.fn.split("/")[:-1]) + "/arbor/arbor.h5"
        
        if self.fn.endswith("arbor.h5"):
            return ytree.load(self.fn)
            
        elif os.path.exists(sfn):
            warnings.warn("h5 formatted arbor has been detected in the provided folder", UserWarning)
            return ytree.load(sfn)
            
        else:
            a = ytree.load(self.fn)
            fn = a.save_arbor(filename=sfn)
            return ytree.load(fn)

    def _compute_Mpeak(self, old_df):
        """Computes the peak mass of each subtree.
        """
        df = deepcopy(old_df)

        df['peak_mass'] = pd.Series()
        unique_subtrees = np.sort(df["Sub_tree_id"].unique())
        for i in tqdm(range(len(unique_subtrees)), desc="Computing Peak Mass of SubTrees", ncols=200):
            sub_tree = unique_subtrees[i]
            sub_df = df[df["Sub_tree_id"] == sub_tree]
            index = sub_df.index
            df.loc[index, "peak_mass"] = sub_df["mass"].values.max()

        return df
        
    def _compute_subtreid(self, old_df, maingal=False):
        """Computes subtree id for given merger-tree tree.
        """
        df = deepcopy(old_df)
        
        df['Sub_tree_id'] = np.zeros(len(df))
        if maingal:
            for snapnum in tqdm(range(self.snap_min, self.snap_max + 1), desc="Computing Sub_tree_id's", leave=False):
                Halo_ID_list = np.unique(df[(df['Snapshot']==snapnum)]['uid'])
                if snapnum == 0:
                    index = df[(df['uid'].isin(Halo_ID_list))].index
                    values = df[(df['uid'].isin(Halo_ID_list))]['uid']
                    df.loc[index, 'Sub_tree_id'] = values
                else:
                    Existing_halos = Halo_ID_list[np.isin(Halo_ID_list, df['desc_uid'])]
                    New_halos = Halo_ID_list[~np.isin(Halo_ID_list, df['desc_uid'])]
                    index_existing = df[(df['uid'].isin(Existing_halos))].sort_values('uid').index
                    index_new = df[(df['uid'].isin(New_halos))].index
                    values_existing = df[(df['desc_uid'].isin(Existing_halos))&
                                                     (df['Secondary']==False)].sort_values('desc_uid')['Sub_tree_id']
                    values_new = df[(df['uid'].isin(New_halos))]['uid']
                    df.loc[index_existing, 'Sub_tree_id'] = np.array(values_existing)
                    df.loc[index_new, 'Sub_tree_id'] = np.array(values_new)
        else:
            for snapnum in range(self.snap_min, self.snap_max + 1):
                Halo_ID_list = np.unique(df[(df['Snapshot']==snapnum)]['uid'])
                if snapnum == 0:
                    index = df[(df['uid'].isin(Halo_ID_list))].index
                    values = df[(df['uid'].isin(Halo_ID_list))]['uid']
                    df.loc[index, 'Sub_tree_id'] = values
                else:
                    Existing_halos = Halo_ID_list[np.isin(Halo_ID_list, df['desc_uid'])]
                    New_halos = Halo_ID_list[~np.isin(Halo_ID_list, df['desc_uid'])]
                    index_existing = df[(df['uid'].isin(Existing_halos))].sort_values('uid').index
                    index_new = df[(df['uid'].isin(New_halos))].index
                    values_existing = df[(df['desc_uid'].isin(Existing_halos))&
                                                     (df['Secondary']==False)].sort_values('desc_uid')['Sub_tree_id']
                    values_new = df[(df['uid'].isin(New_halos))]['uid']
                    df.loc[index_existing, 'Sub_tree_id'] = np.array(values_existing)
                    df.loc[index_new, 'Sub_tree_id'] = np.array(values_new)

        return df
        
    def _compute_host_subtree(self, old_df):
        """Computes the subtree_id of the halo with which it merges later. If not available because of 
        mass cut, we set host_subtree = -1
        """
        df = deepcopy(old_df)
        df_sorted = df.sort_values(['Sub_tree_id', 'Snapshot'])
        last_uids = df_sorted.groupby('Sub_tree_id').tail(1)[['Sub_tree_id', 'desc_uid']]
        
        uid_to_subtree = df.groupby('uid')['Sub_tree_id'].first()
        
        def resolve_host_subtree(desc_uid):
            if desc_uid == -1: return -1
            return uid_to_subtree.get(desc_uid, np.nan)
        
        last_uids['host_subtree'] = last_uids['desc_uid'].apply(resolve_host_subtree)
        
        df = df.merge(last_uids[['Sub_tree_id', 'host_subtree']], on='Sub_tree_id', how='left')
        return df    
        
        
    def _compute_R_Rvir(self, old_df):
        """Computes R/Rvir for given nodes in in the tree.
        """
        df = deepcopy(old_df)

        df['R/Rvir'] = pd.Series()
        df['Rhost'] = pd.Series()
        for snapnum in tqdm(range(self.snap_min, self.snap_max + 1), desc="Computing R/Rvir", ncols=200):
            cx = self.PrincipalLeaf[self.PrincipalLeaf['Snapshot'] == snapnum]['position_x'].values
            cy = self.PrincipalLeaf[self.PrincipalLeaf['Snapshot'] == snapnum]['position_y'].values
            cz = self.PrincipalLeaf[self.PrincipalLeaf['Snapshot'] == snapnum]['position_z'].values
            cRvir = self.PrincipalLeaf[self.PrincipalLeaf['Snapshot'] == snapnum]['virial_radius'].values
            isnap = df[df['Snapshot'] == snapnum].index
            df.loc[isnap, 'R/Rvir'] = np.sqrt(   (df.loc[isnap, 'position_x'] - cx)**2 + 
                                                 (df.loc[isnap, 'position_y'] - cy)**2 + 
                                                 (df.loc[isnap, 'position_z'] - cz)**2 ) / cRvir
            
            df.loc[isnap, "Rhost"] = np.sqrt(   (df.loc[isnap, 'position_x'] - cx)**2 + 
                                                (df.loc[isnap, 'position_y'] - cy)**2 + 
                                                (df.loc[isnap, 'position_z'] - cz)**2 )
            
        return df




    
    def set_fields(self, fields_dict):
        """Sets the fields to be saved into a df
        """
        self.selected_fields = fields_dict
        return None

        
    def set_trees(self, complete_tree):
        """Sets trees by loading the complete tree and separating the principal-leaf, main and satellite-trees. 
        Computations are only even done with the complete-tree. Assumes that "Sub_tree"s and "TreeNum"s are already
        computed. 

        Parameters
        ----------
        complete_tree : str or pd.DataFrame
            Complete tree to be set in place
            
        """
        if isinstance(complete_tree, str):
            self._CompleteTree = pd.read_csv(complete_tree).sort_values("Snapshot")
        elif isinstance(complete_tree, pd.DataFrame):
            self._CompleteTree = complete_tree.sort_values("Snapshot")

        self.snap_min, self.snap_max = int(self.CompleteTree['Snapshot'].values.min()), int(self.CompleteTree['Snapshot'].values.max())

        self.principal_subid, tree_num = self.CompleteTree.sort_values(['mass', 'Snapshot'], ascending = (False, True))[["Sub_tree_id", "TreeNum"]].values[0]
        
        self._PrincipalLeaf = self.CompleteTree[self.CompleteTree["Sub_tree_id"] == self.principal_subid]
        self._MainTree = self.CompleteTree[self.CompleteTree["TreeNum"] == tree_num]
        self._SatelliteTrees = self.CompleteTree[self.CompleteTree["TreeNum"] != tree_num]
        
        self.size = len(self.CompleteTree)


    
    def set_equivalence(self, equiv):
        """Loads and sets equivalence table.
        """
        if isinstance(equiv, str):
            if not equiv.endswith("csv"):
                self._equiv = load_ftable(equiv)
            else:
                self._equiv = pd.read_csv(equiv)
                
        elif isinstance(equiv, pd.DataFrame):
            self._equiv = equiv

        else:
            raise AttributeError("Could not set equivalence table!")

        
    def construc_df_tree(self, treenum, maingal = False):
        """Constructs merger-tree for a single tree.

        OPTIONAL parameters
        -------------------
        treenum : int
            Tree/arbor number inside the forest.s
        """
        if hasattr(self, "_tree_loaded"):
            warnings.warn("What you provided in initialization was an already constructed tree!")
            return None
            

        mytree = self.arbor[treenum]
                
        Data = np.zeros((mytree.tree_size, len(self.selected_fields)))
        single_tree = pd.DataFrame(Data, columns = self.selected_fields.keys())

        if maingal:
            count = 0
            for node in tqdm(list(mytree['tree']), desc="Traversing Main Tree", ncols=200):
                for myname, tablename in self.selected_fields.items():  
                    if type(tablename) == tuple:
                        single_tree.loc[count, myname] = node[tablename[0]].to(tablename[1])
                    else:
                        single_tree.loc[count, myname] = node[tablename]
                count += 1
        else:
            for count, node in enumerate(mytree['tree']):
                for myname, tablename in self.selected_fields.items():  
                    if type(tablename) == tuple:
                        single_tree.loc[count, myname] = node[tablename[0]].to(tablename[1])
                    else:
                        single_tree.loc[count, myname] = node[tablename]


        single_tree_final = single_tree.sort_values(['mass', 'Snapshot'], ascending = (False, True))
        single_tree_final['Secondary'] = single_tree_final.duplicated(['desc_uid', 'Snapshot'], keep='first')
        single_tree_final.sort_values('Snapshot')
        
        if maingal:
            self.snap_min, self.snap_max = int(single_tree_final['Snapshot'].values.min()), int(single_tree_final['Snapshot'].values.max())

        single_tree_final = self._compute_subtreid(single_tree_final)

        if maingal:
            sids = single_tree_final[single_tree_final['Snapshot'] == self.snap_max]['Sub_tree_id'].values
            assert len(sids) == 1, "More than one halo found for the main tree, at z=0!! So Strange!!"
            
            self.principal_subid = int(sids[0])
            self._PrincipalLeaf = single_tree_final[single_tree_final['Sub_tree_id'] == self.principal_subid]
            
        single_tree_final['Halo_at_z0'] = mytree.uid * np.ones_like(single_tree_final['Halo_ID'].values)
        single_tree_final['TreeNum'] =  int(treenum) * np.ones_like(single_tree_final['Halo_ID'].values)

        if self._computing_forest:
            pass
        else:
            single_tree_final = self._compute_R_Rvir(single_tree_final)

        
        return single_tree_final        


    def construct_df_forest(self):
        """Constructs a data-frame based merger-tree forest for easy access
        """
        if hasattr(self, "_tree_loaded"):
            warnings.warn("What you provided in initialization was an already constructed tree!")
            return None
            
        self._computing_forest = True
        
        MainTree = self.construc_df_tree(0, maingal=True) 
        MainTree = self._compute_R_Rvir(MainTree)
        MainTree = self._compute_Mpeak(MainTree)

        z0_masses = np.array([tree['mass'].to(self.selected_fields['mass'][1]).value/self.min_halo_mass - 1 for tree in self.arbor[1:]])
        index_above = np.where((z0_masses > 0) == False)[0][0] + 1

        SatelliteTrees_final = pd.DataFrame()
        for sat_index in tqdm(range(1, index_above), desc="Traversing Satellite Trees", ncols=200):
            SatTree = self.construc_df_tree(sat_index, maingal=False)
            SatelliteTrees_final = pd.concat([SatelliteTrees_final, SatTree])

        SatelliteTrees_final = self._compute_R_Rvir(SatelliteTrees_final.reset_index(drop=True, inplace=False))
        SatelliteTrees_final = self._compute_Mpeak(SatelliteTrees_final)

        # self._MainTree = MainTree
        # self._SatelliteTrees = SatelliteTrees_final
        CompleteTree = pd.concat([MainTree, SatelliteTrees_final])
        CompleteTree = self._compute_host_subtree(CompleteTree)
        self.set_trees(CompleteTree)

        self._computing_forest = False

        return None


    def postprocess(self, subtree_ids):
        """Postprocesses some of the trees to determine with which halo it ends up merging.
        """
        #SelectedTree ---<-<-<->--->->-> Tree_massive
        #CompleteTree ---<-<-<->---->->-> Tree
        CompleteTree = deepcopy(self.CompleteTree)
        SelectedTree = CompleteTree[CompleteTree['Sub_tree_id'].isin(subtree_ids)]
        
        id_merge = ()
        snap_merge_arr = ()
        snap_crossing_arr = ()
        snap_crossing_arr_2 = ()
        sub_tree_id_host_arr = ()
        Ratio_mass_arr = ()
        Mass_crossing_arr = ()
        virial_pos_merge = ()
        peak_mass_arr = ()
        
        spike_tree_id = ()
        snap_spike = ()
        birth_snap = ()
        birth_distance = ()
        desc_arr = ()

        n = len(SelectedTree["Sub_tree_id"].unique())
        if not "host_distance" in CompleteTree.columns:
            CompleteTree['host_distance'] = np.zeros( len(CompleteTree) )
            
        for i in tqdm(range(n), desc="Running over Subtrees", ncols=100):
            subtree = SelectedTree["Sub_tree_id"].unique()[i]
            SubTree = SelectedTree[SelectedTree['Sub_tree_id'] == subtree].sort_values('Snapshot', ascending=False)

            faulty = False
            
            if (SubTree['mass'].diff(periods=-1) / SubTree["mass"]).max() > 0.8:
                faulty = True
                snap_max = SubTree[(SubTree['mass'].diff(periods=-1) / SubTree["mass"]) == (SubTree['mass'].diff(periods=-1) / SubTree["mass"]).max()]["Snapshot"].iloc[0]

                spike_tree_id = np.append(spike_tree_id, subtree)
                snap_spike = np.append(snap_spike, snap_max)

                
                if len(SubTree[SubTree['Secondary'] == True]["desc_uid"]) > 0:
                    desc = SubTree[SubTree["Secondary"] == True]["desc_uid"].iloc[0]
                else:
                    desc = np.nan
                desc_arr = np.append(desc_arr, desc)

            birth = SubTree["Snapshot"].min()
            birth_rvir = SubTree[SubTree["Snapshot"] == birth]["R/Rvir"].iloc[0]
            if faulty:
                birth_snap = np.append(birth_snap, birth)
                birth_distance = np.append(birth_distance, birth_rvir)
                
            elif birth_rvir < 2:
                birth_snap = np.append(birth_snap, birth)
                birth_distance = np.append(birth_distance, birth_rvir)
                spike_tree_id = np.append(spike_tree_id, subtree)
                snap_spike = np.append(snap_spike, np.nan)
                if len(SubTree[SubTree['Secondary'] == True]["desc_uid"]) > 0:
                    desc = SubTree[SubTree["Secondary"] == True]["desc_uid"].iloc[0]
                else:
                    desc = np.nan
                desc_arr = np.append(desc_arr, desc)
        

            
            if len(SubTree[SubTree['Secondary'] == True]) > 0:
                merger_index = SubTree[SubTree['Secondary'] == True].index[0]
                uid_host = SubTree.loc[merger_index, 'desc_uid']
                subtree_host = CompleteTree[(CompleteTree['uid'] == uid_host)]['Sub_tree_id'].iloc[0]
                merger_snap = SubTree.loc[merger_index, 'Snapshot']
            else:
                subtree_host = self.principal_subid
                merger_snap = self.snap_max

            index_subtree = SubTree.index
            snap_start = SubTree['Snapshot'].min()

            HostTree = CompleteTree[(CompleteTree['Sub_tree_id'] == subtree_host) & (CompleteTree['Snapshot'] <= merger_snap) &
                            (CompleteTree['Snapshot'] >= snap_start)].sort_values('Snapshot', ascending=False)
            HostTree_start = HostTree['Snapshot'].min()
        
            position_central = HostTree[['position_x', 'position_y', 'position_z']].values   
            position_sub = SubTree[['position_x', 'position_y', 'position_z']].values            
            rvir_central = HostTree['virial_radius']

            CompleteTree.loc[index_subtree, 'Sub_tree_id_host'] = subtree_host

            id_merge = np.append(id_merge, subtree)
            snap_merge_arr = np.append(snap_merge_arr, merger_snap)
            
            if (HostTree_start > snap_start):
                Tree_sub_exist = SubTree[SubTree['Snapshot'] >= HostTree_start]
                index_subtree = Tree_sub_exist.index
                index_nan = SubTree[SubTree['Snapshot'] < HostTree_start].index
                position_sub = Tree_sub_exist[['position_x', 'position_y', 'position_z']].values   
                CompleteTree.loc[index_nan, 'host_distance'] = np.nan
            else:
                index_subtree = index_subtree
                
            CompleteTree.loc[index_subtree, 'host_distance'] = np.array( np.sqrt( np.sum( (position_sub - position_central)**2, axis=1) ) / rvir_central )
            
            
            if len(CompleteTree[(CompleteTree['Sub_tree_id'] == subtree) & (CompleteTree['host_distance'] > 1)].sort_values('Snapshot', ascending=True)['Snapshot']) > 0:
                snap_crossing = CompleteTree[(CompleteTree["Sub_tree_id"] == subtree) & (CompleteTree["host_distance"] > 1)].sort_values("Snapshot", ascending=False)["Snapshot"].iloc[0]
            else:
                snap_crossing = merger_snap

            if len(CompleteTree[(CompleteTree['Sub_tree_id'] == subtree) & (CompleteTree['host_distance'] > 2)].sort_values('Snapshot', ascending=True)['Snapshot']) > 0:
                    snap_crossing_2 = CompleteTree[(CompleteTree["Sub_tree_id"] == subtree) & (CompleteTree["host_distance"] > 2)].sort_values("Snapshot", ascending=False)["Snapshot"].iloc[0]
            else:
                snap_crossing_2 = merger_snap


            Ratio_mass = SubTree[SubTree['Snapshot'] == snap_crossing]['mass'].iloc[0] / HostTree[HostTree['Snapshot'] == snap_crossing]['mass'].iloc[0]
            Mass_crossing = SubTree[SubTree['Snapshot'] == snap_crossing]['mass'].iloc[0] 
            RRvir_merge = SubTree[SubTree['Snapshot'] == snap_crossing]['R/Rvir'].iloc[0] 
            peak_M = SubTree[SubTree['Snapshot'] == snap_crossing]['peak_mass'].iloc[0]
            
            snap_crossing_arr = np.append(snap_crossing_arr, snap_crossing)
            snap_crossing_arr_2 = np.append(snap_crossing_arr_2, snap_crossing_2)
            sub_tree_id_host_arr = np.append(sub_tree_id_host_arr, subtree_host)
            Ratio_mass_arr = np.append(Ratio_mass_arr, Ratio_mass)
            Mass_crossing_arr = np.append(Mass_crossing_arr, Mass_crossing)
            peak_mass_arr = np.append(peak_mass_arr, peak_M)

            virial_pos_merge = np.append(virial_pos_merge, RRvir_merge)



        d = {
            'Snapshot': snap_merge_arr, 'Sub_tree_id': id_merge, 'Sub_tree_id_host': sub_tree_id_host_arr, 'crossing_snap' : snap_crossing_arr,
            'crossing_snap_2': snap_crossing_arr_2, "R/Rvir_merge": virial_pos_merge, 'mass_ratio' : Ratio_mass_arr, 'crossing_mass' : Mass_crossing_arr, "peak_mass": peak_mass_arr
        }
        s = {
            'Sub_tree_id': spike_tree_id, 'Snapshot_spike': snap_spike, "birth_snapshot": birth_snap, "birth_R/Rvir": birth_distance, 'final_desc_uid': desc_arr
        }
        
        if hasattr(self, "MergeInfo"):
            self.MergeInfo = pd.concat([self.MergeInfo, pd.DataFrame(data = d).sort_values("Sub_tree_id")])
            self.SpikeInfo = pd.concat([self.MergeInfo, pd.DataFrame(data = s).sort_values("Sub_tree_id")])
        else:
            self.MergeInfo = pd.DataFrame(data = d).sort_values("Sub_tree_id")
            self.SpikeInfo = pd.DataFrame(data = s).sort_values("Sub_tree_id")

        self.set_trees(CompleteTree)

    def thinn_catalogue(self, thinned_snapshot):
        """Provides a way to thinn the equivalence table and MergerTrees to some snapshot_values 
        if the user doesn't have access to all the snapshots. Thinning to very coarse time steps might 
        lead to untrackability of halos.
        """
        new_merge = MergerTree(self._CompleteTree[self._CompleteTree["Snapshot"].isin(thinned_snapshot)].sort_values("Snapshot").reset_index(drop=True))
        new_merge.set_equivalence(self._equiv[self._equiv["snapshot"].isin(thinned_snapshot)].reset_index(drop=True)) 

        if hasattr(self, "MergeInfo"):
            new_merge.MergeInfo = self.MergeInfo

            new_merge.MergeInfo['Snapshot'] = new_merge.MergeInfo['Snapshot'].transform(lambda x: max([s for s in thinned_snapshot if s <= x], default=min(thinned_snapshot)))
            new_merge.MergeInfo['crossing_snap'] = new_merge.MergeInfo['crossing_snap'].transform(lambda x: max([s for s in thinned_snapshot if s <= x], default=min(thinned_snapshot)))
            new_merge.MergeInfo['crossing_snap_2'] = new_merge.MergeInfo['crossing_snap_2'].transform(lambda x: max([s for s in thinned_snapshot if s <= x], default=min(thinned_snapshot)))
            
            new_merge.SpikeInfo = self.SpikeInfo
            
        return new_merge


    def subtree(self, subtree):
        """Returns the given subtree
        """
        return self.CompleteTree[self.CompleteTree["Sub_tree_id"] == subtree]


    def select_halos(self, Rvir=1, **constraints):
        """Selects halos according to the provided constraints. Constraints are pased as kwargs. Sub_tree_id == 1 is avoided.

        Parameters
        ----------
        Rvir : float
            First radius in which to search. Default: R/Rvir=1.
        extra_Rvirs : array[float]
            Extra R/Rvir in which to search for the halos found at Rvir.
        constraints : kwargs
            Constraints to apply at Rvir. 

            List:
            ------------
            Redshift : [zlow, zmax]
            mass: [mlow, mhigh]
            Secondary : True or False
            Rvir_tol : 0.2

            Constraints on stellar mass must be performed afterwards, as the computation of bound stellar particles is not straitgforward.
            Furthermore, that would require particle data.
            After constraining in stellar_mass, one can trace-back the halos to more extreme R/Rvir with the 'traceback_halos' method.

        Returns
        -------
        crossing_haloes : pd.DataFrame
            Haloes crossing Rvir and fullfilling constraints.
        """
        self.main_Rvir = Rvir
        zlow, zhigh, mlow, mhigh, keep_secondary, Rvir_tol = -1, 1E4, -1, 1E20, True, 0.2
        keep_ocurrences = "first"
        
        CompleteTree = deepcopy(self.CompleteTree)
        if "peak_mass" not in CompleteTree.columns:
            CompleteTree = self._compute_Mpeak(CompleteTree)
            
        if 'redshift' in constraints.keys():
            zlow, zhigh = constraints['redshift']
        if 'mass' in constraints.keys():
            mlow, mhigh = constraints['mass']
        if 'keep_secondary' in constraints.keys():
            keep_secondary = constraints['keep_secondary']
        if 'Rvir_tol' in constraints.keys():
           Rvir_tol = constraints['Rvir_tol']
        if 'keep_ocurrences' in constraints.keys():
            keep_ocurrences = constraints["keep_ocurrences"]    


        constrainedTree_Rmain = CompleteTree[(np.abs(CompleteTree['R/Rvir'] - Rvir) < Rvir_tol) & 
                                             (mlow <= CompleteTree['peak_mass'])                     & (CompleteTree['peak_mass'] <= mhigh) &
                                             (zlow <= CompleteTree['Redshift'])                      & (CompleteTree['Redshift'] <= zhigh)  &
                                             ((CompleteTree['Secondary'] == False)                   | (CompleteTree['Secondary'] == keep_secondary))
                                            ].sort_values(by=['Snapshot', 'Sub_tree_id'], ascending=[True, True])

        constrainedTree_Rmain['Priority'] = (constrainedTree_Rmain['R/Rvir'] > Rvir).astype(int)  
        constrainedTree_Rmain['R_diff'] = np.abs(constrainedTree_Rmain['R/Rvir'] - Rvir)

        constrainedTree_Rmain.loc[constrainedTree_Rmain.index,'crossings'] = (
            constrainedTree_Rmain.groupby('Sub_tree_id')['Priority']
            .transform(lambda x: (x != x.shift()).cumsum() - 1)
        )
        
        constrainedTree_Rmain_sorted = constrainedTree_Rmain.sort_values(
            by=['Sub_tree_id', 'Priority', 'crossings', 'R_diff'],
            ascending=[True, False, True, True]
        )
        
        #selected_halos = constrainedTree_Rmain_sorted.groupby('Sub_tree_id').first().reset_index()

        if keep_ocurrences == 'first':
            selected_halos = constrainedTree_Rmain_sorted.groupby('Sub_tree_id').first().reset_index()
        elif keep_ocurrences == 'last':
            selected_halos = constrainedTree_Rmain_sorted.groupby('Sub_tree_id').last().reset_index()
        elif keep_ocurrences == 'all':
            selected_halos = constrainedTree_Rmain_sorted.copy()
        else:
            raise ValueError("Invalid value for keep_ocurrences: must be 'first', 'last', or 'all'")

        
        crossing_haloes_mainRvir = selected_halos.drop(['crossings', 'Priority', 'R_diff'], axis=1)
        
        return crossing_haloes_mainRvir, constrainedTree_Rmain


    def traceback_halos(self, Rvirs, halodf):
        """Traces back selected halos to different Rvirs. This method accompanies select_halos: this selects halos at a given Rvir and according to certain
        constraints on mass, radshift, merging histories etc. traceback_halos traces back those halos to more outer R/Rvir radii.
        
        It is a requirement that Rvirs > Rvir.

        Parameters
        ----------
        Rvirs : list[float]
            Rvirs to traceback.
        halodf : list[int]
            Halo dataframe created with select_halos.

        Returns
        -------
        
        """
        self.extra_Rvirs = np.array(Rvirs)
        if np.any(self.extra_Rvirs <= self.main_Rvir):
            raise Exception("All extra R/Rvir must be greater than the main Rvir.")

        concated = np.append(self.extra_Rvirs, self.main_Rvir)
        concated = np.sort(np.append(concated, np.max(self.extra_Rvirs)*2))
        lower = 0.5*(concated[:-1] - np.roll(concated, -1)[:-1])
        upper = 0.5*(concated[1:] - np.roll(concated, 1)[1:])
        rvir_bounds = {v : [v+l, v+u] for v, l, u in zip(concated[1:-1], lower, upper[1:])}
        
        CompleteTree = deepcopy(self._CompleteTree)
        
        if np.unique(halodf['Sub_tree_id'].values).shape != halodf['Sub_tree_id'].values.shape:
            raise Exception("Subtree shapes are fucked up")
        else:
            subtree_redshifts = {sid : z for sid, z in zip(halodf['Sub_tree_id'].values, halodf['Redshift'].values)}
            
        dataframes = {}
        selected_halos = pd.DataFrame()
        
        for rvir, bounds in rvir_bounds.items():
            constrainedTree_rvir = CompleteTree[CompleteTree['Sub_tree_id'].isin( list(subtree_redshifts.keys()) )]
            
            constrainedTree_rvir = constrainedTree_rvir[(constrainedTree_rvir['R/Rvir'] >= bounds[0]) & 
                                                        (constrainedTree_rvir['R/Rvir'] <= bounds[1]) 
                                                       ]

            constrainedTree_rvir = constrainedTree_rvir.merge(pd.DataFrame({'Sub_tree_id': list(subtree_redshifts.keys()), 'reference_z': list(subtree_redshifts.values())}),
                                                              on='Sub_tree_id',
                                                              how='left'
                                                             )

            constrainedTree_rvir = constrainedTree_rvir[constrainedTree_rvir['Redshift'] >= constrainedTree_rvir['reference_z']]
            
            constrainedTree_rvir['Priority'] = (constrainedTree_rvir['R/Rvir'] > rvir).astype(int)  

            constrainedTree_rvir['R_diff'] = np.abs(constrainedTree_rvir['R/Rvir'] - rvir)
            
            constrainedTree_rvir = constrainedTree_rvir.sort_values(by=['Sub_tree_id', 'Redshift'], ascending=[True, False])
            
            constrainedTree_rvir['crossings'] = (
                constrainedTree_rvir.groupby('Sub_tree_id')['Priority']
                .transform(lambda x: (x != x.shift()).cumsum() - 1)
            )

            
            sorted_tree_top = constrainedTree_rvir.sort_values(by=['Sub_tree_id', 'Priority', 'crossings', 'R_diff'], ascending=[True, False, False, True])
            sorted_tree_bottom = constrainedTree_rvir.sort_values(by=['Sub_tree_id', 'Priority', 'crossings', 'R_diff'], ascending=[True, True, False, True])

            selected_at_target_top = sorted_tree_top.groupby('Sub_tree_id').first().reset_index()
            selected_at_target_bottom = sorted_tree_bottom.groupby('Sub_tree_id').first().reset_index()

            all_subtrees = np.unique(
                np.concatenate( (selected_at_target_top['Sub_tree_id'].values, selected_at_target_bottom['Sub_tree_id'].values) )
            )
            selected_at_target = pd.DataFrame()
            for subtree in all_subtrees:
                in_top = selected_at_target_top[selected_at_target_top['Sub_tree_id'] == subtree]
                in_bottom = selected_at_target_bottom[selected_at_target_bottom['Sub_tree_id'] == subtree]
    
                if (not in_top.empty) and (not in_bottom.empty):
                    if in_top['R_diff'].iloc[0] <= in_bottom['R_diff'].iloc[0]:
                        selected_at_target = pd.concat([selected_at_target, in_top])
                    else:
                        selected_at_target = pd.concat([selected_at_target, in_bottom])
                elif not in_top.empty:
                    selected_at_target = pd.concat([selected_at_target, in_top])
                elif not in_bottom.empty:
                    selected_at_target = pd.concat([selected_at_target, in_bottom])

            selected_at_target = selected_at_target.reset_index(drop=True)
            selected_at_target['target_R/Rvir'] = rvir
            
            selected_halos = pd.concat([selected_halos, selected_at_target])

        non_monotonic_mask = selected_halos.groupby('Sub_tree_id')['Redshift'].apply(lambda x: x != x.sort_values().cummax())
        selected_halos = selected_halos[~non_monotonic_mask.values]

        dataframes = {rvir_value: df.drop(columns=['R_diff', 'target_R/Rvir', 'crossings'], inplace=False)
                      for rvir_value, df in selected_halos.groupby('target_R/Rvir')}
        return dataframes


    def infall_from(self, subtree_list, Rvir, **constraints):
        """Returns a dataframe that contains all the entries
        """
        zlow, zhigh, mlow, mhigh= -1, 1E4, -1, 1E20
        keep_ocurrences = "last"
        
        CompleteTree = deepcopy(self.CompleteTree)
        if "peak_mass" not in CompleteTree.columns:
            CompleteTree = self._compute_Mpeak(CompleteTree)
            
        if 'redshift' in constraints.keys():
            zlow, zhigh = constraints['redshift']
        if 'mass' in constraints.keys():
            mlow, mhigh = constraints['mass']
        if 'keep_ocurrences' in constraints.keys():
            keep_ocurrences = constraints["keep_ocurrences"]   

        mask = (
            self.CompleteTree["Sub_tree_id"].isin(subtree_list) & 
            (self.CompleteTree["Redshift"] <= zhigh)
        )
    
        df = deepcopy(self.CompleteTree[mask].sort_values("Snapshot", ascending=True))
    
        df['Priority'] = (df['R/Rvir'] > Rvir).astype(int)  
        
        df['R_diff'] = np.abs(df['R/Rvir'] - Rvir)
        
        df = df.sort_values(by=['Sub_tree_id', 'Redshift'], ascending=[True, False])
        
        df['crossings'] = (
            df.groupby('Sub_tree_id')['Priority']
            .transform(lambda x: (x != x.shift()).cumsum() - 1)
        )
        if keep_ocurrences == "last":
            df['ref_crossings'] = df.groupby('Sub_tree_id')['crossings'].transform('max')
        elif keep_ocurrences == "first":
            df['ref_crossings'] = df.groupby('Sub_tree_id')['crossings'].transform('min') + 1

        satellites = df[df['crossings'] >= df['ref_crossings']].sort_values("Snapshot", ascending=True).copy()        
        satellites.drop(["Priority", "R_diff", "crossings", "ref_crossings"], axis=1, inplace=True)
        return satellites

    
    def save(self, code=""):
        """Saves Main, Satellite, Complete and Host merger trees
        """
        if code == "":
            self.MainTree.to_csv("MainTree.csv", index=False)
            self.SatelliteTrees.to_csv("SatelliteTrees.csv", index=False)
            self.CompleteTree.to_csv("CompleteTree.csv", index=False)
            self.PrincipalLeaf.to_csv("HostTree.csv", index=False)
        else:
            self.MainTree.to_csv(f"{code}_MainTree.csv", index=False)
            self.SatelliteTrees.to_csv(f"{code}_SatelliteTrees.csv", index=False)
            self.CompleteTree.to_csv(f"{code}_CompleteTree.csv", index=False)
            self.PrincipalLeaf.to_csv(f"{code}_HostTree.csv", index=False)

        return None

    
    def find_file(self, snapshot):
        """Finds the file corresponding to a given snapshot.
        """
        result = self.equivalence_table[self.equivalence_table['snapshot'].values == snapshot]['snapname'].values
        if len(result) > 0:
            return result[0]
        else:
            raise ValueError(f"Snapshot {snapshot} doesn't exist!")

            
    def set_snapshot(self, path=None, dataset=None):
        """Loads a snapshot, by path, dataset.
        """
        if dataset is not None:
            self._ds = dataset
        elif path is not None:
            self._ds = config.loader(path)

        
    def get_halo_params(self, sub_tree, redshift=None, snapshot=None):
        """Loads a halo identified by its sub_tree ID using the Halo class. A given redshift or snapshot number can be
        suplied to load the halo at a single redshift, or load all of its snapshots.
        """
        closest_value = lambda number, array: array[np.argmin(np.abs(np.array(array) - number))]

        if self.equivalence_table is None:
            raise ValueError("There is no snapshot number <----> snapshot file equivalence table set!")

        skip = False
        if snapshot == -1:
            index = self.CompleteTree[
                (self.CompleteTree["Sub_tree_id"] == sub_tree) & 
                (self.CompleteTree["Snapshot"] == self.CompleteTree["Snapshot"].max())
            ].index
            skip = True
        if redshift is None and not skip:
            index = self.CompleteTree[
                (self.CompleteTree["Sub_tree_id"] == sub_tree) & 
                (self.CompleteTree["Snapshot"] == snapshot)
            ].index
        elif snapshot is None:
            index = self.CompleteTree[
                (self.CompleteTree["Sub_tree_id"] == sub_tree) & 
                (self.CompleteTree["Redshift"] == closest_value(redshift, self.CompleteTree["Redshift"].values))
            ].index

        
        halo = self.CompleteTree.loc[index]
        halo_params = {
            "redshift": halo["Redshift"].values[0].astype(float),
            "center": unyt_array(halo[["position_x", "position_y", "position_z"]].values[0].astype(float) / (1 + halo["Redshift"].values[0].astype(float)), 'kpc'),
            "center_vel": unyt_array(halo[["velocity_x", "velocity_y", "velocity_z"]].values[0].astype(float), 'km/s'),
            "rvir": unyt_quantity(halo["virial_radius"].values[0].astype(float) / (1 + halo["Redshift"].values[0].astype(float)), 'kpc'),
            "rs": unyt_quantity(halo["scale_radius"].values[0].astype(float) / (1 + halo["Redshift"].values[0].astype(float)), 'kpc'),
            "vmax": unyt_quantity(halo["vmax"].values[0].astype(float), 'km/s'),
            "vrms": unyt_quantity(halo["vrms"].values[0].astype(float), 'km/s'),
            "mass": unyt_quantity(halo["mass"].values[0].astype(float), 'Msun'),
            "time" : unyt_quantity(halo["Time"].values[0].astype(float), 'Gyr'),
            "R/Rvir" : halo["R/Rvir"].values[0].astype(float),
        }
        return halo_params

        
    def load_halo(self, sub_tree, redshift=None, snapshot=None, particle_ids=None, gas_ids=None):
        """Loads a halo identified by its sub_tree ID using the Halo class. A given redshift or snapshot number can be
        suplied to load the halo at a single redshift, or load all of its snapshots.
        """
        halo = self.get_halo_params(sub_tree, redshift=redshift, snapshot=snapshot)
                    
        self.ad = self.ds.all_data()
        self.sp = self.ds.sphere(halo["center"], halo["rvir"])

        if particle_ids is not None:
            stars = DataContainer(self.ad, "stars", particle_ids=particle_ids)
            dm = DataContainer(self.ad, "darkmatter", particle_ids=particle_ids)
        else:
            stars = DataContainer(self.sp, "stars")
            dm = DataContainer(self.sp, "darkmatter")            

            
        if gas_ids is not None:
            gas = DataContainer(self.sp, "gas")
        else:
            gas = DataContainer(self.sp, "gas")
            
        return [stars, dm, gas]

    









#constrainedTree_Rmain_sorted = constrainedTree_Rmain.sort_values(by=['Sub_tree_id', 'Priority', 'Redshift', 'R_diff'], 
                                                         #ascending=[True, False, False, True]
                                                        #)
#crossing_haloes_mainRvir = constrainedTree_Rmain_sorted.groupby('Sub_tree_id').first().reset_index().drop(["R_diff", 'Priority'], axis=1)




#constrainedTree_rvir['R_diff'] = np.abs(constrainedTree_rvir['R/Rvir'] - rvir)
#constrainedTree_rvir = constrainedTree_rvir.merge(pd.DataFrame({'Sub_tree_id': list(subtree_redshifts.keys()), 'reference_z': list(subtree_redshifts.values())}),
#                                                  on='Sub_tree_id',
#                                                  how='left'
#                                                 )
#constrainedTree_rvir['z_diff'] = constrainedTree_rvir['Redshift'] - constrainedTree_rvir['reference_z']
#constrainedTree_rvir = constrainedTree_rvir[constrainedTree_rvir['z_diff'] >= 0]
#constrainedTree_rvir_sorted = constrainedTree_rvir.sort_values(by=['Sub_tree_id', 'R_diff', 'z_diff'], 
#                                                               ascending=[True, True, True]
#                                                              )
#crossing_haloes_rvir = constrainedTree_rvir_sorted.groupby('Sub_tree_id').first().reset_index().drop(columns=['R_diff', 'z_diff', 'reference_z'])
#
#dataframes[rvir] = crossing_haloes_rvir

#return selected_halos



#df = mt
#df = df.reset_index(drop=True)
#idx = df.groupby('Snapshot')['mass'].idxmax()

#result = df.loc[idx].reset_index(drop=True)    combine result into one df
    
