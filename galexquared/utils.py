import yt
import numpy as np
from pytreegrav import PotentialTarget

from .class_methods import compute_stars_in_halo, create_sph_dataset



def particle_unbinding_fire(data_source,
                            ptype,
                            halo_params,
                            max_radius=(30, 'kpc'), 
                            nmin=6,
                            imax=200,
                            verbose=False,
                            **kwargs
                           ):
    """Returns a mask of particle "boundness" following the method described in one of the 
    FIREbox papers. Note that energies are not computed at any time, rather, cuts on position
    and velocity are imposed based on halo rotation curve moments vrms and vmax.
    """
    pos = data_source[ptype, "particle_position"].to("kpc")
    vels = data_source[ptype, "particle_velocity"].to("km/s")
    masses = data_source[ptype, "particle_mass"].to("Msun")
    pindices = data_source[ptype, "particle_index"]
    return compute_stars_in_halo(
        pos,
        masses,
        vels,
        pindices,
        halo_params,
        max_radius=max_radius, 
        nmin=nmin,
        imax=imax,
        verbose=verbose,
        **kwargs
    )


def quickplot(data_containers, normal="z", catalogue=None, smooth_particles=False, verbose=False, **kwargs):
    """Plots the projected darkmatter, stars and gas distributions along a given normal direction. The projection direction can be changed with
    the "normal" argument. If catalogue is provided, halos of massgreater than 5E7 will be displayed on top of the darkmatter plot.

    OPTIONAL Parameters
    ----------
    normal : str or 
        Projection line of sight, either "x", "y" or "z".
    catalogue : pd.DataFrame
        Halo catalogue created using galex.extractor.mergertrees.

    Returns
    -------
    fig : matplotlib figure object
    """
    import cmyt
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Circle
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset
    
    try:
        import smplotlib
    except:
        pass

    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['xtick.major.width'] = 1.1
    plt.rcParams['xtick.minor.width'] = 1.1
    plt.rcParams['ytick.major.width'] = 1.1
    plt.rcParams['ytick.minor.width'] = 1.1
    
    plt.rcParams['xtick.major.size'] = 7 * 1.5
    plt.rcParams['ytick.major.size'] = 7 * 1.5
    
    plt.rcParams['xtick.minor.size'] = 5 
    plt.rcParams['ytick.minor.size'] = 5
    
    if verbose: print(f"Extracting data containers...")

    ds = data_containers[0].ds
    center = data_containers[0].yt_region.center
    radius = data_containers[0].yt_region.radius

    stars_dc = data_containers[0]
    dm_dc = data_containers[1]
    gas_dc = data_containers[2]

    if verbose: print(f"Looking for kwargs: zoom_factors, catalogue, smoothing...")

    source_factor = 1.5 if "source_factor" not in kwargs.keys() else kwargs["source_factor"]
    draw_inset = True if "draw_inset" not in kwargs.keys() else kwargs["draw_inset"]
    
    if "zoom_factors_rvir" in kwargs.keys():
        zooms = np.sort(kwargs["zoom_factors_rvir"])[::-1][[0,1]]
        plot_radii = radius * zooms * 2
    else:
        plot_radii = radius * np.array([1 * 2])

    sp_source = ds.sphere(center, source_factor * plot_radii.max())
    
    plots = 0
    if not stars_dc.empty(): plots += 1
    if not dm_dc.empty(): plots += 1
    if not gas_dc.empty(): plots += 1

    if plots == 0:
        raise ValueError("It Seems that all components are empty!")
    if verbose: print(f"Total plots: {plots}...")

    if normal=="x" or normal==[1,0,0]: cindex = [1,2]
    elif normal=="y" or normal==[0,1,0]: cindex = [2,0]
    elif normal=="z" or normal==[0,0,1]: cindex = [0,1]
    else: raise ValueError(f"Normal must along main axes. You provided {normal}.")

    if smooth_particles:
        if verbose: print(f"Smoothing particles...")

        ds_dm = create_sph_dataset(
            ds,
            "darkmatter",
            data_source=sp_source,
            n_neighbours=100 if "n_neighbours" not in kwargs.keys() else kwargs["n_neighbours"][0] if isinstance(kwargs["n_neighbours"], list) else kwargs["n_neighbours"],
            kernel="wendland2" if "kernel" not in kwargs.keys() else kwargs["kernel"],
        )
        dm_type = ("io", "density") 

        ds_st = create_sph_dataset(
            ds,
            "stars",
            data_source=sp_source,
            n_neighbours=100 if "n_neighbours" not in kwargs.keys() else kwargs["n_neighbours"][1] if isinstance(kwargs["n_neighbours"], list) else kwargs["n_neighbours"],
            kernel="wendland2" if "kernel" not in kwargs.keys() else kwargs["kernel"],
        )
        star_type = ("io", "density")

        sp_source_dm = ds_dm.sphere(center, source_factor*plot_radii.max())
        sp_source_st = ds_st.sphere(center, source_factor*plot_radii.max())
    else:
        dm_type = ("darkmatter", "mass")  
        star_type = ("stars", "mass")            

    gas_type = ("gas", "density")        

    if verbose: print(f"Setting colormap...")

    if "cm" not in kwargs:
        cmyt.arbre.set_bad(cmyt.arbre.get_under())
        cm = cmyt.arbre
    else:
        cm = kwargs["cm"]


    if verbose: print(f"Start plotting...")

    fig, axes = plt.subplots(
        len(plot_radii),
        plots,
        figsize=(6*plots*1.2*1.05, 6*len(plot_radii)*1.2),
        sharey="row",
        constrained_layout=False
    )
    grid = axes.flatten()

    stars_cen = stars_dc["cm"]
    dm_cen = dm_dc["cm"]
    gas_cen = gas_dc["cm"]
    plot_centers = [center, 0.5 * (stars_cen + dm_cen)]
    
    
    ip = 0
    for jp, (pcenter, plot_radius) in enumerate(zip(plot_centers, plot_radii)):
        center_dist =  (pcenter - plot_centers[0])[cindex]

        ext = (-plot_radius.to("kpc")/2 + center_dist[0], 
               plot_radius.to("kpc")/2 + center_dist[0], 
               -plot_radius.to("kpc")/2 + center_dist[1], 
               plot_radius.to("kpc")/2 + center_dist[1]
        )



        tmp_stars_cen = (stars_cen - pcenter)[cindex] + center_dist
        tmp_dm_cen = (dm_cen - pcenter)[cindex] + center_dist
        tmp_gas_cen = (gas_cen - pcenter)[cindex] + center_dist
        tmp_cen = (center - pcenter)[cindex] + center_dist

        if not dm_dc.empty(): 
            if smooth_particles:            
                p = yt.ProjectionPlot(ds_dm, normal, dm_type, center=pcenter, width=plot_radius, data_source=sp_source_dm)
                p.set_unit(dm_type, "Msun/kpc**2")
                frb = p.data_source.to_frb(plot_radius, 800)

            else:
                p = yt.ParticleProjectionPlot(ds, normal, dm_type, center=pcenter, width=plot_radius, density=True, data_source=sp_source, deposition="ngp" if "deposition" not in kwargs.keys() else kwargs["deposition"])
                p.set_unit(dm_type, "Msun/kpc**2")
                frb = p.frb


            ax = grid[ip]
            data = frb[dm_type]
            data[data == 0] = data[data != 0 ].min()
            pc_dm = ax.imshow(data.to("Msun/kpc**2"), cmap=cm, norm="log", extent=ext)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0)
            cbar_dm = plt.colorbar(pc_dm, cax=cax)
            cbar_dm.set_label(r'Projected Dark Matter Density $[Msun/kpc^2]$', fontsize=20)
            cbar_dm.ax.tick_params(labelsize=25)

            
            ax.scatter(*tmp_cen, color="black", marker="1", s=370, zorder=20)
            if jp==0:
                rvir_circ = Circle((0,0), radius.to("kpc").value, facecolor="none", edgecolor="black")
                ax.add_patch(rvir_circ)
                ax.scatter(*tmp_stars_cen, color="red", marker="*", s=150)
                ax.scatter(*tmp_dm_cen, color="black", marker="+", s=300)
            if jp==1:
                ax.scatter(*tmp_stars_cen, color="red", marker="*", s=300)
                ax.scatter(*tmp_dm_cen, color="black", marker="+", s=370)
                ax.scatter(*tmp_gas_cen, color="orange", marker="o")
                rhf_circ = Circle(tmp_stars_cen.to("kpc").value, stars_dc.half_mass_radius().to("kpc").value, facecolor="none", edgecolor="red")
                ax.add_patch(rhf_circ)
            
            ax.set_aspect('equal')
            ax.tick_params(axis='both', which='both', labelbottom=True, labelsize=20)    
            if ip < plots:
                ax.set_title(kwargs["titles"][0] if "titles" in kwargs.keys() else None, fontsize=25)

            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            ip += 1

        
        if not stars_dc.empty():
            if smooth_particles: 
                p = yt.ProjectionPlot(ds_st, normal, star_type, center=pcenter, width=plot_radius, data_source=sp_source_st)
                p.set_unit(star_type, "Msun/kpc**2")
                frb = p.data_source.to_frb(plot_radius, 800)

            else:
                p = yt.ParticleProjectionPlot(ds, normal, star_type, center=pcenter, width=plot_radius, density=True, data_source=sp_source, deposition="ngp" if "deposition" not in kwargs.keys() else kwargs["deposition"])
                p.set_unit(star_type, "Msun/kpc**2")    
                frb = p.frb


            ax = grid[ip]
            data = frb[star_type]
            data[data == 0] = data[data != 0 ].min()
            pc_st = ax.imshow(data.to("Msun/kpc**2"), cmap=cm, norm="log", vmin=data.to("Msun/kpc**2").max().value/1E4, extent=ext)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0)
            cbar_st = fig.colorbar(pc_st, cax=cax)
            cbar_st.set_label(r'Projected Stellar Density $[Msun/kpc^2]$', fontsize=22)
            cbar_st.ax.tick_params(labelsize=25)

            ax.scatter(*tmp_cen, color="black", marker="1", s=370, zorder=20)
            if jp==0:
                rvir_circ = Circle((0,0), radius.to("kpc").value, facecolor="none", edgecolor="black")
                ax.add_patch(rvir_circ)
                ax.scatter(*tmp_stars_cen, color="red", marker="*", s=150)
                ax.scatter(*tmp_dm_cen, color="black", marker="+", s=300)
            if jp==1:
                ax.scatter(*tmp_stars_cen, color="red", marker="*", s=300)
                ax.scatter(*tmp_dm_cen, color="black", marker="+", s=370)
                ax.scatter(*tmp_gas_cen, color="orange", marker="o")
                rhf_circ = Circle(tmp_stars_cen.to("kpc").value, stars_dc.half_mass_radius().to("kpc").value, facecolor="none", edgecolor="red")
                ax.add_patch(rhf_circ)


            
            ax.set_aspect('equal')
            ax.tick_params(axis='both', which='both', labelbottom=True, labelsize=20)
            if ip < plots:
                ax.set_title(kwargs["titles"][1] if "titles" in kwargs.keys() else None, fontsize=25)

            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])               
            ip += 1

        
        if not gas_dc.empty(): 
            p = yt.ProjectionPlot(ds, normal, gas_type, center=pcenter, width=plot_radius, data_source=sp_source)
            p.set_unit(gas_type, "Msun/kpc**2")
            frb = p.data_source.to_frb(plot_radius, 800)
            
            ax = grid[ip]
            density_data = frb[gas_type]
            density_data[density_data == 0] = density_data[density_data != 0 ].min()
            p_gas = ax.imshow(density_data.to("Msun/kpc**2"), cmap=cm, norm="log", extent=ext)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0)
            cbar_gas = fig.colorbar(p_gas, cax=cax)
            cbar_gas.set_label(r'Projected Gas Density $[Msun/kpc^2]$', fontsize=22)

            ax.scatter(*tmp_cen, color="black", marker="1", s=370, zorder=20)
            if jp==0:
                rvir_circ = Circle((0,0), radius.to("kpc").value, facecolor="none", edgecolor="black")
                ax.add_patch(rvir_circ)
                ax.scatter(*tmp_stars_cen, color="red", marker="*", s=150)
                ax.scatter(*tmp_dm_cen, color="black", marker="+", s=300)
                ax.scatter(*tmp_gas_cen, color="orange", marker="o")
            if jp==1:
                ax.scatter(*tmp_stars_cen, color="red", marker="*", s=300)
                ax.scatter(*tmp_dm_cen, color="black", marker="+", s=370)
                ax.scatter(*tmp_gas_cen, color="orange", marker="o")
                rhf_circ = Circle(tmp_stars_cen.to("kpc").value, stars_dc.half_mass_radius().to("kpc").value, facecolor="none", edgecolor="red")
                ax.add_patch(rhf_circ)

            
            cbar_gas.ax.tick_params(labelsize=25)
            ax.set_aspect('equal')
            ax.tick_params(axis='both', which='both', labelbottom=True, labelsize=20)
            if ip < plots:
                ax.set_title(kwargs["titles"][2] if "titles" in kwargs.keys() else None, fontsize=25)

            ax.set_xlim(ext[0], ext[1])
            ax.set_ylim(ext[2], ext[3])
            ip += 1

    c = "white" if smooth_particles else "darkgreen"
    low_m = 9E7 if "low_mass" not in kwargs.keys() else kwargs["low_mass"]
    high_m = np.inf if "high_mass" not in kwargs.keys() else kwargs["high_mass"]
    annotation_style = "circle" if "annotation_style" not in kwargs.keys() else kwargs["annotation_style"]
    if catalogue is not None:
        dist = np.linalg.norm(ds.arr(catalogue[['position_x', 'position_y', 'position_z']].values, 'kpccm') - center.to("kpccm"), axis=1)
        filtered_halos = catalogue[
            (dist < plot_radii.max()) & 
            (catalogue['mass'] > low_m) & 
            (catalogue['mass'] < high_m) & 
            (dist > 0.1)
        ]
        for i in range(0, len(filtered_halos)):
            sub_tree_id = filtered_halos['Sub_tree_id'].iloc[i]
            halo_pos = ds.arr(filtered_halos.iloc[i][['position_x', 'position_y', 'position_z']].values, 'kpccm').to('kpc') - center
            virial_radius = ds.quan(filtered_halos.iloc[i]['virial_radius'], 'kpccm').to('kpc')

            if annotation_style == "circle" or annotation_style == "all":
                extra_halo = Circle(halo_pos[cindex], 0.5*virial_radius, facecolor="none", edgecolor=c)
                axes[0, 0].add_patch(extra_halo)
                
            if annotation_style == "center" or annotation_style == "all":
                axes[0, 0].scatter(*halo_pos[cindex], marker="v", edgecolor=c, s=90, color="none")

            if annotation_style == "all":
                axes[0, 0].text(halo_pos[cindex][0], halo_pos[cindex][1] - 0.033*virial_radius, int(sub_tree_id), fontsize=14, ha="center", va="top", color=c)    

        

    for ax in axes[-1,:]:
        ax.set_xlabel('x [kpc]', fontsize=20)

    for ax in axes[:, 0]:
        ax.set_ylabel('y [kpc]', fontsize=20)

    if len(plot_radii) != 1 and draw_inset:
        mark_inset(
            axes[0, -1], 
            axes[-1, -1], 
            loc1=1, loc2=2, 
            facecolor="none",
            edgecolor="black"
        )

        plt.subplots_adjust(
            hspace=-0.45,
            wspace=0.1
        )
    else:
        plt.subplots_adjust(
            hspace=0.3,
            wspace=0.1
        )
    plt.tight_layout()
    plt.close()
    return fig  






