import yt
import argparse
import numpy as np
import pandas as pd

import gala.dynamics as gd
import astropy.units as u

from scipy.signal import savgol_filter, argrelextrema, argrelmax, argrelmin, peak_prominences
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import smplotlib

import galexquared as gal
from galexquared.class_methods import load_ftable
from galexquared import refind_pericenters_apocenters


def parse_args():
    parser = argparse.ArgumentParser(description="Script to interpolate orbits using Orbis.")
        
    required = parser.add_argument_group('REQUIRED arguments')
    opt = parser.add_argument_group('OPTIONAL arguments')
    
    
    required.add_argument(
        "-i", "--input_file",
        type=str,
        help="Merger Tree file path. Must be csv. Tree must be constructed in csv format already.",
        required=True
    )
    required.add_argument(
        "-s", "--subtree_ids",
        type=str,
        help="Sub_tree_ids to analyze.",
        required=True
    )
    required.add_argument(
        "-o", "--output",
        type=str,
        help="Location of particle data. Requred for selecting starry halos.",
        required=True
    )
    required.add_argument(
        "-c", "--code",
        type=str,
        help="Code of the simulation. Requred for selecting starry halos.",
        required=True
    )    


    opt.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Verbose output. (Default False)"
    )
    opt.add_argument(
        "--number",
        nargs="*",
        type=int,
        default=[1,1],
        help="Pericenter - apocenter combination to select"
    )
    opt.add_argument(
        "--orbit_folder", "-of",
        type=str,
        default=None,
        help="Folde where the interpolated orbits live.",
    ) 
    opt.add_argument(
        "-sf", "--simulation_folder",
        type=str,
        help="Location of simulation data.",
        default=None
    )
    opt.add_argument(
        "-pf", "--particles_folder",
        type=str,
        help="Location of the tracked particles.",
        default=None
    )
    opt.add_argument(
        "-eq", "--equivalence_table",
        type=str,
        help="Equivalence table path. Must have correct formatting.",
        default=None
    )
    opt.add_argument(
        "--stellar_particle",
        type=str,
        help="Stellar particle name.",
        default=None
    )
    return parser.parse_args()


def find_closest_times(df, timestamp):
    """
    Finds the closest "Time" value to the given timestamp, along with the previous and next "Time" values.

    Parameters:
        df (pd.DataFrame): DataFrame containing a "Time" column (datetime format).
        timestamp (str or pd.Timestamp): The timestamp to find the closest match for.

    Returns:
        dict: A dictionary containing 'closest_time', 'previous_time', and 'next_time'.
    """
    # Ensure "Time" column is in datetime format
    df = df.copy().reset_index(drop=True)

    # Find the closest time index
    idx_closest = (df['Time'] - timestamp).abs().idxmin()
    # Find previous and next indices safely
    idx_prev = idx_closest - 1 if idx_closest > 0 else idx_closest
    idx_next = idx_closest + 1 if idx_closest < len(df) - 1 else idx_closest
    
    return df.loc[idx_prev, 'Time'], df.loc[idx_closest, 'Time'], df.loc[idx_next, 'Time']

def find_bracketing_snaps(df, timestamp):
    """
    Finds the snapshot numbers just before and just after a given timestamp.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'Time' and 'Snapshot' columns.
        timestamp (float or pd.Timestamp): The timestamp to bracket.

    Returns:
        tuple: (snapshot_before, snapshot_after) — if no before/after exists, returns None in that position.
    """
    df = df.copy().sort_values("Time").reset_index(drop=True)

    before_df = df[df["Time"] < timestamp]
    after_df = df[df["Time"] > timestamp]

    snapshot_before = before_df.iloc[-1]["Snapshot"] if not before_df.empty else None
    snapshot_after = after_df.iloc[0]["Snapshot"] if not after_df.empty else None
    
    time_before = before_df.iloc[-1]["Time"] if not before_df.empty else None
    time_after = after_df.iloc[0]["Time"] if not after_df.empty else None
    return snapshot_before, snapshot_after, time_before, time_after

def extract_stellar_mass(ds, ptype, indices):
    """Computes bound stellar mass as given by Ramon Tracking.
    """
    yt.add_particle_filter(
        "bound_stars", 
        function=lambda pfilter, data: np.isin(data[pfilter.filtered_type, "particle_index"], indices), 
        filtered_type=ptype, 
        requires=["particle_index", "particle_mass"]
    )
    ds.add_particle_filter("bound_stars")
    ad = ds.all_data()
    mass = ad["bound_stars", "particle_mass"].to("Msun").sum()
    ad.clear_data()
    del ds, ad
    return mass



args = parse_args()


gal.config.code = args.code

MT = gal.MergerTree(args.input_file)
MT.set_equivalence(args.equivalence_table)
pdir = args.simulation_folder
tdir = args.particles_folder
candidates = pd.read_csv(args.subtree_ids)


units = [u.Gyr, u.kpc, u.Msun, u.deg, u.km/u.s]


if args.code == "ART":
    subtree_list = [213, 236, 239, 262, 277, 987, 7788,
                    14642, 14757, 23533, 29557, 29952, 30302, 32830,
                    32806, 34701, 40092, 40554, 40578, 43679,
                    53092, 54928, 56181, 58973, 59565, 59693,
                    67126, 90652, 104106, 151846, 162050,
                    167667, 220710, 239354, 490284
                   ]
                   
if args.code == "GEAR":
    subtree_list = [410, 451, 899, 1059, 1071, 1199, 1209, 1274, 1304, 1311,
                    1463, 1473, 17608, 28974, 31608, 42065, 50789, 77801,
                    92193, 165594, 183069, 186762, 214096, 312003, 
                    346178,  620179, 4273404
                   ]
    equiv = load_ftable(args.equivalence_table)
    equiv["snapshot"] = equiv.index



full_table = pd.DataFrame(columns=["Sub_tree_id", "Type", "Redshift", "Time", "radius", "dm_mass", "stellar_mass", "detection"])


nc = np.ceil(np.sqrt(len(subtree_list))).astype(int)
nr = int(len(subtree_list) / nc) // 2 * 2 + 1

if nr*nc < len(subtree_list):
    nr += 1

fig, axes = plt.subplots(
    nrows=nr,
    ncols=nc,
    figsize=(nr * 8, nc * 5)
)

axesf = axes.flatten()

for i, subtree in enumerate(subtree_list):
    print(f"\n\n SUBTREE {subtree}")
    ts = MT.infall_from([subtree], 1.5, keep_ocurrences="first")

    subtree_orbit = pd.read_csv(f"output/orbit_interpolation_{gal.config.code}/Interpolated_orbit_{int(subtree)}.csv")
    
    sub_orbit = gd.Orbit(
        subtree_orbit[["position_x", "position_y", "position_z"]].values.T * u.kpc,
        subtree_orbit[["velocity_x", "velocity_y", "velocity_z"]].values.T * u.km/u.s,
        t=subtree_orbit["Time"].values * u.Gyr
    )
    ts["Rhost_kpc"] =  ts["Rhost"].values / (1 + ts["Redshift"].values)
    pa, pa_all = refind_pericenters_apocenters(ts["Time"].values, ts["Rhost_kpc"], verbose=1)
    print("\n\n")
    subtree_table = {
        "Sub_tree_id": [],
        "Type": [],
        "Redshift": [],
        "Time": [],
        "radius": [],
        "dm_mass": [],
        "stellar_mass": [],
        "detection": []
    }
    for apocenter in pa["apo"]:
        time_prev, time_closest, time_next = find_closest_times(ts, apocenter["time"])
        subtree_table["Sub_tree_id"].append(int(subtree))
        subtree_table["Type"].append("apo")

        mask = (subtree_orbit["Time"].values <= time_next) & (time_prev <= subtree_orbit["Time"].values)
        times = sub_orbit.t.value[mask]
        radii = np.linalg.norm(sub_orbit.xyz.T, axis=1).value[mask]
        
        apo_index = np.nanargmax(radii)

        subtree_table["Time"].append(times[apo_index])
        subtree_table["radius"].append(radii[apo_index])
        subtree_table["Redshift"].append(np.nan)
        subtree_table["detection"].append(apocenter["detection"])

        axesf[i].plot(times, radii, color="magenta", ls="--", zorder=9)
        axesf[i].scatter(times[apo_index], radii[apo_index], color="magenta", ls="--", marker="*", zorder=9)



        snap_before, snap_after, time_before, time_after = find_bracketing_snaps(ts, times[apo_index])
        file_before, file_after = MT.equivalence_table[MT.equivalence_table["snapshot"] == snap_before]["snapname"].values[0], MT.equivalence_table[MT.equivalence_table["snapshot"] == snap_after]["snapname"].values[0]
        particles_before, particles_after = np.load(tdir + f"subtree_{int(subtree)}/Particles_{float(subtree)}_{int(snap_before)}.npy"), np.load(tdir + f"subtree_{int(subtree)}/Particles_{float(subtree)}_{int(snap_after)}.npy")
        ds_before, ds_after = yt.load(pdir + "/" + file_before), yt.load(pdir + "/" + file_after)

        mass_before = extract_stellar_mass(ds_before, args.stellar_particle, particles_before)
        mass_after = extract_stellar_mass(ds_after, args.stellar_particle, particles_after)

        st_mass_at_apo = np.interp(times[apo_index], [time_before, time_after], [mass_before.value, mass_after.value])
        dm_mass_at_apo = np.interp(times[apo_index], ts["Time"].values, ts["mass"].values)

        
        subtree_table["stellar_mass"].append(st_mass_at_apo)
        subtree_table["dm_mass"].append(dm_mass_at_apo)


        
        
    for pericenter in pa["peri"]:
        time_prev, time_closest, time_next = find_closest_times(ts, pericenter["time"])
        subtree_table["Sub_tree_id"].append(int(subtree))
        subtree_table["Type"].append("peri")

        mask = (subtree_orbit["Time"].values <= time_next) & (time_prev <= subtree_orbit["Time"].values)
        times = sub_orbit.t.value[mask]
        radii = np.linalg.norm(sub_orbit.xyz.T, axis=1).value[mask]

        peri_index = np.nanargmin(radii)

        subtree_table["Time"].append(times[peri_index])
        subtree_table["radius"].append(radii[peri_index])
        subtree_table["Redshift"].append(np.nan)
        subtree_table["detection"].append(pericenter["detection"])

        axesf[i].plot(times, radii, color="magenta", ls="--", zorder=9)
        axesf[i].scatter(times[peri_index], radii[peri_index], color="magenta", ls="--", marker="*", zorder=9)



        snap_before, snap_after, time_before, time_after = find_bracketing_snaps(ts, times[peri_index])
        file_before, file_after = MT.equivalence_table[MT.equivalence_table["snapshot"] == snap_before]["snapname"].values[0], MT.equivalence_table[MT.equivalence_table["snapshot"] == snap_after]["snapname"].values[0]
        particles_before, particles_after = np.load(tdir + f"subtree_{int(subtree)}/Particles_{float(subtree)}_{int(snap_before)}.npy"), np.load(tdir + f"subtree_{int(subtree)}/Particles_{float(subtree)}_{int(snap_after)}.npy")
        ds_before, ds_after = yt.load(pdir + "/" + file_before), yt.load(pdir + "/" + file_after)

        mass_before = extract_stellar_mass(ds_before, args.stellar_particle, particles_before)
        mass_after = extract_stellar_mass(ds_after, args.stellar_particle, particles_after)

        st_mass_at_peri = np.interp(times[peri_index], [time_before, time_after], [mass_before, mass_after])
        dm_mass_at_peri = np.interp(times[peri_index], ts["Time"].values, ts["mass"].values)

        subtree_table["stellar_mass"].append(st_mass_at_peri)
        subtree_table["dm_mass"].append(dm_mass_at_peri)
    

    subtree_df = pd.DataFrame.from_dict(subtree_table)
    full_table = pd.concat([full_table, subtree_df])

    print(subtree_df)
    
    axesf[i].scatter(ts["Time"],  ts["Rhost"].values / (1 + ts["Redshift"].values), s=20)
    axesf[i].plot(sub_orbit.t, np.linalg.norm(sub_orbit.xyz.T, axis=1), color="blue", ls="--")

    for value in pa_all["apo"]:
        axesf[i].scatter(value["time"],value["radius"] , color="red", s=20)
        tmp = np.linspace(value["time"] - 1, value["time"] + 1, 1000)

    for value in pa_all["peri"]:
        axesf[i].scatter(value["time"],value["radius"] , color="blue", s=20)
        tmp = np.linspace(value["time"] - 1, value["time"] + 1, 1000)

    for value in pa["apo"]:
        axesf[i].scatter(value["time"],value["radius"] , color="green", s=20)
        tmp = np.linspace(value["time"] - 1, value["time"] + 1, 1000)

    for value in pa["peri"]:
        axesf[i].scatter(value["time"],value["radius"] , color="purple", s=30)
        tmp = np.linspace(value["time"] - 1, value["time"] + 1, 1000)

        
    axesf[i].set_yscale("log")

    x0, x1 = axesf[i].get_xlim()
    _, y1 = axesf[i].get_ylim()
    axesf[i].text(0.5 * (x0 + x1), 0.95 * y1, f"{args.code}-{int(subtree)}", transform=axesf[i].transData, ha="center", va="top")



first_pa_table = pd.DataFrame(columns=["Sub_tree_id", "Type", "Redshift", "Time", "radius", "detection"])

for subtree in subtree_list:
    subtree_data = full_table[full_table["Sub_tree_id"] == subtree]

    first_apo = subtree_data[subtree_data["Type"] == "apo"].sort_values("Time").head(1)
    first_peri = subtree_data[subtree_data["Type"] == "peri"].sort_values("Time").head(1)

    if not first_apo.empty:
        first_pa_table = pd.concat([first_pa_table, first_apo])
    if not first_peri.empty:
        first_pa_table = pd.concat([first_pa_table, first_peri])



for i, subtree in enumerate(subtree_list):
    sub = first_pa_table[first_pa_table["Sub_tree_id"] == subtree]
    axesf[i].scatter(sub["Time"], sub["radius"], color="darkorange", marker="*", zorder=9)


































































for ax in axes[-1, :]:
    ax.set_xlabel("Time [Gyr]")
for ax in axes[:, 0]:
    ax.set_ylabel("r [kpc]")

fig.suptitle(f"Infall orbits of AGORA {args.code} Satellites", fontsize=50, y=0.91)

plt.savefig(f"{args.output}/{args.code}_infall-pa.png")
plt.close()


full_table.to_csv(f"{args.output}/{args.code}_infall-pa.csv", index=False)
first_pa_table.to_csv(f"{args.output}/{args.code}_first_pa.csv", index=False)

