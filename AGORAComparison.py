import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
from astropy.table import Table
from cmdstanpy import CmdStanModel
from matplotlib import pyplot as plt
import astropy.units as u

from tqdm import tqdm
import smplotlib

def parse_args():
    parser = argparse.ArgumentParser(description="Script to plot dwarf galaxy properties in different codes along their evolution/infall")
        
    required = parser.add_argument_group('REQUIRED arguments')
    opt = parser.add_argument_group('OPTIONAL arguments')
    
    opt.add_argument(
        "-c", "--config",
        type=str,
        help="YAML config file",
        default=None,
        required=False
    )
    
    opt.add_argument(
        "-i", "--input_file",
        nargs="*",
        type=str,
        help="Input file(s) for simulated dwarf properties",
        required=False
    )
    opt.add_argument(
        "-s", "--subtree_ids",
        nargs="*",
        type=int,
        help="Sub_tree_ids to plot.",
        required=False
    )
    opt.add_argument(
        "-cd", "--comparison_data",
        type=str,
        help="Input file(s) for real dwarf properties",
        required=False
    )
    opt.add_argument(
        "-o", "--output",
        type=str,
        help="Location of output plots",
        required=False
    )
    
    opt.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Verbose output. (Default False)"
    )
    return parser.parse_args()


def get_yaml_data(file_path):
    """ Read data from yaml file
    """
    with open(file_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
        
    return yaml_data 
    

def mag_to_lum(mags):
    global Msun
    return 10 ** ( (mags - Msun)/(-2.5) )



    
args = parse_args()



Msun = 4.74

if bool(args.config):
    config = get_yaml_data(args.config)

    output_dir = config["output"]
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        
    files_subtrees = dict(zip( config["dwarfs"]["files"], config["dwarfs"]["subtrees"] ))
    combine = config["files_in_one"]
    plot_names = config["dwarfs"]["alias"]
    lit_dir = config["literature"]
    pltstyle = config["plot"]


mccon = Table.read(f'{lit_dir}/LocalGroup/McConnachie_properties_complete.dat', format = "ascii.fixed_width", delimiter="\t")
mccon2 = Table.read(f'{lit_dir}/LocalGroup/McCOnnachie_2012.csv')
solo_dwarfs = Table.read(f'{lit_dir}/LocalGroup/SoloDwarfs_IV_complete.dat', format = "ascii.fixed_width", delimiter="\t")
globc = Table.read(f"{lit_dir}/GC_Baumgardt2018.fits", format="fits")



mccon2["ML"] = 1E6 * mccon2["Mdyn"] / (0.5 * mag_to_lum(mccon2["VMag"]))

#globc = Table.read(f"{lit_dir}/GC_Harris1997.dat", format="ascii.csv",delimiter=";", comment="#")

#globc["rh"]


s4g_sample = Table.read(f"{lit_dir}/cs4g_selection_leda.fits", format="fits")


s4g_sample["re"] = 0.5 * (s4g_sample["re1"] +  s4g_sample["re2"]) + 0.5 * (s4g_sample["re3"] +  s4g_sample["re4"]) 
s4g_sample["R2"] = s4g_sample["re"].to(u.radian).value * s4g_sample["dist"].to("pc")

s4g_sample["Mdyn"] = (580*s4g_sample["R2"].to("pc")*s4g_sample["vdis"].to("km/s")**2).value * u.Msun
s4g_sample["VMag"] = s4g_sample["mabs"] - s4g_sample["bvtc"]




my_model = CmdStanModel(stan_file="./scripts/linear_noErrors_robust.stan", cpp_options={'STAN_THREADS': 'true'})

dwarf_data = {
"N": len(mccon['VMag'].value),
"M": 2,
"x": np.log10(mccon['Mdyn'].value*1E6).tolist(),
"y": mccon['VMag'].value.tolist(),
'xeval' : np.log10([10,10]).tolist()
}

fit = my_model.sample(data=dwarf_data, chains=4, iter_sampling=5000, show_console=False)

df = fit.draws_pd()

Msun = 4.74




if pltstyle == "all" or pltstyle == "dynamical":
    if combine:
        k = -1
        for file, subtrees in files_subtrees.items():  
            k += 1
            name = plot_names[k]
    
            morph_mccon = mccon2
            
            trans_dict = {'dIrr/dSph': "p",
                          'dIrr': "*",
                          'dE/dSph': "h",
                          'cE': "D",
                          "Sc": "P",
                          'Irr': "s", 
                          '????': "x",
                          '(d)Irr?': "s",
                          'dSph' : "o",
                          'dSph?': "o"
                         }
            
            import matplotlib.gridspec as gridspec
            
            
            
            plt.rcParams['axes.linewidth'] = 1.1
            plt.rcParams['xtick.major.width'] = 1.1
            plt.rcParams['xtick.minor.width'] = 1.1
            plt.rcParams['ytick.major.width'] = 1.1
            plt.rcParams['ytick.minor.width'] = 1.1
            
            plt.rcParams['xtick.major.size'] = 7
            plt.rcParams['ytick.major.size'] = 7
            
            
            rh_lim = (1, 1E4)
            MV_lim = (2, -24)
            Mdyn_lim = (1E4, 9E10)
            sigma_lim = (1, 200)
            
            
            z = np.polyfit(np.log10(mccon['Mdyn'] * 1E6), np.log10(mccon['LV'] * 1E6 * 0.5), 1)
            p = np.poly1d(z)
            md = np.linspace(*Mdyn_lim, 1000)
            lhalf = 10 ** (z[0] * np.log10(md) + z[1])
            MV = 4.84 - 2.5 * np.log10(2*lhalf)
            
            
            fig = plt.figure(figsize=(1.5*1.8*9,1.8*9))
                
                
            gs = gridspec.GridSpec(2, 3, hspace=0, wspace=0)
            
            ax1 = fig.add_subplot(gs[0])  
            ax2 = fig.add_subplot(gs[1])  
            ax3 = fig.add_subplot(gs[4])  
            ax4 = fig.add_subplot(gs[3])  
            
            ax5 = fig.add_subplot(gs[2])  
            ax6 = fig.add_subplot(gs[5])  
            
            
            j = 0
            
            i = 0
            for key, value in trans_dict.items():
                morph_mccon = mccon2[mccon2['MType'] == key]
                
                if key != "dSph?" and key != "(d)Irr?":
                    ax1.scatter([np.nan],[np.nan], marker=value, color="black", s=25, label=f"{key}") 
            
                    
                if i==8:
                    a1 = ax1.scatter(morph_mccon['R2'][morph_mccon['SubG'] == "MW"], morph_mccon['VMag'][morph_mccon['SubG'] == "MW"], marker=value, color="red", s=25, label=f"Milky-Way group")
                    a2 = ax1.scatter(morph_mccon['R2'][morph_mccon['SubG'] == "M31"], morph_mccon['VMag'][morph_mccon['SubG'] == "M31"], marker=value, color="blue",s=25, label=f"Andromeda gruop")
                    a3 = ax1.scatter(morph_mccon['R2'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], morph_mccon['VMag'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], marker=value, color="green", s=25, label=f"Local group "+r"($D_{LG} \leq 1Mpc$)")
                    a4 = ax1.scatter(morph_mccon['R2'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], morph_mccon['VMag'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], marker=value, color="magenta", s=25, label=f"Nearby galaxies "+r"($D_{LG} > 1Mpc$)")
                else:
                    ax1.scatter(morph_mccon['R2'][morph_mccon['SubG'] == "MW"], morph_mccon['VMag'][morph_mccon['SubG'] == "MW"], marker=value, color="red", s=25 )
                    ax1.scatter(morph_mccon['R2'][morph_mccon['SubG'] == "M31"], morph_mccon['VMag'][morph_mccon['SubG'] == "M31"], marker=value, color="blue",s=25 )
                    ax1.scatter(morph_mccon['R2'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], morph_mccon['VMag'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], marker=value, color="green", s=25 )
                    ax1.scatter(morph_mccon['R2'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], morph_mccon['VMag'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], marker=value, color="magenta", s=25 ) 
            
            
                
                ax1.set_xscale("log")
                ax1.invert_yaxis()
                ax1.set_xlim(*rh_lim)
                ax1.set_ylim(*MV_lim)
                ax1.set_xlabel(r"$r_h$ [pc]")
                ax1.set_xticklabels([0, 1, 10, 100, 1000, 10000])
                ax1.set_ylabel(r"$M_V$ [mag]" )
                ax1.legend()
            
                i += 1
                
                
                
                
                y = morph_mccon['R2'] 
                x = morph_mccon['Mdyn'] * 1E6 
                ax3.scatter(x[morph_mccon['SubG'] == "MW"], y[morph_mccon['SubG'] == "MW"], marker=value, color="red", s=25, )
                ax3.scatter(x[morph_mccon['SubG'] == "M31"], y[morph_mccon['SubG'] == "M31"], marker=value, color="blue",s=25, )
                ax3.scatter(1E6 * morph_mccon['Mdyn'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], morph_mccon['R2'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], marker=value, color="green", s=25, )
                ax3.scatter(1E6 * morph_mccon['Mdyn'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], morph_mccon['R2'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], marker=value, color="magenta", s=25, )
                
                ax3.set_ylabel(r"$r_h$ [pc]")
                ax3.set_xlabel(r"$M_{dyn} (\leq r_h) \ \ [M_\odot]$")
                ax3.loglog()
                ax3.set_xlim(*Mdyn_lim)
                ax3.set_ylim(ax1.get_xlim())
                ax3.set_yticklabels(ax1.get_xticklabels())
                
                
                
                
                
                
                y = morph_mccon['VMag']
                x = 1E6 * morph_mccon['Mdyn']
                ax2.scatter(x[morph_mccon['SubG'] == "MW"], y[morph_mccon['SubG'] == "MW"], marker=value, color="red", s=25, )
                ax2.scatter(x[morph_mccon['SubG'] == "M31"], y[morph_mccon['SubG'] == "M31"], marker=value, color="blue",s=25, )
                ax2.scatter(x[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], y[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], marker=value, color="green", s=25, )
                ax2.scatter(x[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], y[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], marker=value, color="magenta", s=25, )
                
                
                 
                
                
                ax2.set_xscale("log")
                ax2.invert_yaxis()
                ax2.set_ylim(ax1.get_ylim())
                ax2.set_ylabel(None)
                ax2.set_yticklabels([])
                ax2.set_xticklabels([])
                ax2.set_xlabel(None)
                ax2.set_xlim(ax3.get_xlim())
            
            
            
                x = morph_mccon['sigma*']
                y = morph_mccon['R2']
                ax6.scatter(x[morph_mccon['SubG'] == "MW"], y[morph_mccon['SubG'] == "MW"], marker=value, s=25, color="red")
                ax6.scatter(x[morph_mccon['SubG'] == "M31"], y[morph_mccon['SubG'] == "M31"],  marker=value, s=25, color="blue")
                ax6.scatter(x[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], y[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], marker=value, color="green", s=25, )
                ax6.scatter(x[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], y[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], marker=value, color="magenta", s=25, )
                
                ax6.set_xscale("log")
                ax6.set_yscale("log")
                ax6.set_xlim(*sigma_lim)
                ax6.set_xlabel(r"$\sigma_*$ [km/s]")
                ax6.set_ylabel("")
                ax6.set_yticklabels([])
                ax6.set_ylim(ax3.get_ylim())
            
            
            
            
            
                x = morph_mccon['sigma*']
                y = morph_mccon['VMag']
                ax5.scatter(x[morph_mccon['SubG'] == "MW"], y[morph_mccon['SubG'] == "MW"], marker=value, s=25, color="red")
                ax5.scatter(x[morph_mccon['SubG'] == "M31"], y[morph_mccon['SubG'] == "M31"],  marker=value, s=25, color="blue")
                ax5.scatter(x[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], y[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], marker=value, color="green", s=25, )
                ax5.scatter(x[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], y[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], marker=value, color="magenta", s=25, )    
                
                
                ax5.set_xscale("log")
                ax5.set_yscale("linear")
                ax5.set_xlim(ax6.get_xlim())
                ax5.set_xlabel("")
                ax5.set_ylabel("")
                ax5.set_yticklabels([])
                ax5.set_xticklabels([])
                ax5.set_ylim(ax1.get_ylim())
            
            
            legend_GC = ax1.scatter(globc["rh"], globc['MVt'], s=10, label="MW globular clusters")
            ax2.scatter(globc["Mdyn"], globc['MVt'], s=10)
            ax3.scatter(globc["Mdyn"], globc['rh'], s=10)
            ax6.scatter(globc["sigma0"], globc['rh'], s=10)
            ax5.scatter(globc["sigma0"], globc['MVt'], s=10)
    
            
            bf = ax2.plot(md, df['beta0'].mean() + df['beta1'].mean()*np.log10(md), lw=2, zorder=10, label="Best Fit")
            
            for beta0, beta1 in df[['beta0', 'beta1']].values[::1]:
                ax2.plot(md, beta0 + beta1 * np.log10(md), lw=1, alpha=0.1, color="lightblue", zorder=-1)
            
            
            
            trans_dict_s4g = {
                'E': "o",
                'E-S0': "p",
                'S0': "d",
                'S0-a': "d",
                "SABc": "s",
                'SBa': "x", 
                'SBb': "x",
                'SBbc': "x",
                'Sbc' : "P",
                'Sc': "P",
                'Sm': "P"
            }
            labels_s4g = {
                'E': "E ",
                'E-S0': "E-S0 ",
                'S0': "S0 ",
                "SABc": "SAB ",
                'SBa': "SB ", 
                'Sm': "S "
            }
            
            for key, value in trans_dict_s4g.items():
                morph_s4g = s4g_sample[s4g_sample['type'] == key]
            
                if key in ["E", "S0", "E-S0", "SABc", "SBa", "Sm"]:
                    ax5.scatter([np.nan],[np.nan], marker=value, color="tomato", s=25, label=f"{labels_s4g[key]}") 
            
                b1 = ax1.scatter(morph_s4g["R2"], morph_s4g["VMag"], marker=value, color="tomato", s=25)
                b1 = ax3.scatter(morph_s4g["Mdyn"], morph_s4g["R2"], marker=value, color="tomato", s=25)
                b1 = ax2.scatter(morph_s4g["Mdyn"], morph_s4g["VMag"], marker=value, color="tomato", s=25)
                b1 = ax6.scatter(morph_s4g["vdis"], morph_s4g["R2"], marker=value, color="tomato", s=25)
                b1 = ax5.scatter(morph_s4g["vdis"], morph_s4g["VMag"], marker=value, color="tomato", s=25)
                
                    
            table = pd.read_csv(file)
            grouped_table = table[table['Sub_tree_id'].isin(subtrees)].groupby("Sub_tree_id")
            
            for (subid, candidate_df) in grouped_table:
                candidate_df = candidate_df.sort_values("Snapshot", ascending=True)
                n = len(candidate_df)
                
                print(f"{subid} df. shape: {n}")
                print(candidate_df)
                
                
                rrvir = candidate_df['R/Rvir'].values
                targets = [rrvir.min(), 0.5, 1, 2, 3, 4]
                indices = [np.abs(rrvir - target).argmin() for target in targets]
                rrvir_markers = ["*", "^", "^", "^", "^", "o"]
                
                MV_1 = Msun - 2.5 * np.log10(candidate_df['LV'])
                
                smooth = lambda x: np.convolve(x, np.ones(5) / 5, mode='same')
                
                if MV_1.values.shape != smooth(MV_1.values).shape: raise Exception(f"Smoothing went wrong.")
    
    
                smooth_MV = smooth(MV_1.values)
                smooth_rh = smooth(candidate_df['rh_stars_physical'].values)
                smooth_mdyn = smooth(candidate_df['Mdyn'].values)
                smoth_sigma = smooth(candidate_df['sigma*'].values)
                
                ax1.plot(smooth_rh * 1E3, smooth_MV, color="black")
                ax6.plot(smoth_sigma , smooth_rh * 1E3,  color="black")   
                ax5.plot(smoth_sigma , smooth_MV,  color="black")
                ax3.plot(smooth_mdyn, smooth_rh * 1E3,  color="black")
                ax2.plot(smooth_mdyn, smooth_MV, color="black")
                
                
                for index, marker in zip(indices, rrvir_markers): 
                    ax1.scatter(smooth_rh[index] * 1E3, smooth_MV[index], color="gray", marker=marker, s=15)
                    ax6.scatter(smoth_sigma[index] , smooth_rh[index] * 1E3,  color="gray", marker=marker, s=15)   
                    ax5.scatter(smoth_sigma[index] ,smooth_MV[index],  color="gray", marker=marker, s=15)
                    ax3.scatter(smooth_mdyn[index], smooth_rh[index] * 1E3,  color="gray", marker=marker, s=15)
                    ax2.scatter(smooth_mdyn[index], smooth_MV[index], color="gray", marker=marker, s=15)
                
                ax1.text(smooth_rh[-1]*1E3, 0 if subid != 5888 else -0.75, str(int(subid)), ha="center", va="top", fontsize=10
                         , color="gray")
                ax1.scatter(smooth_rh[-1]*1E3, -0.25, marker="^"
                         , color="gray")
                ax6.text(smoth_sigma[-1], 1.44 if subid != 236 else 2, str(int(subid)), ha="center", va="top", fontsize=10
                         , color="gray")
                ax6.scatter(smoth_sigma[-1], 1.6, marker="^"
                         , color="gray")
                ax3.text(smooth_mdyn[-1], 1.44 if subid != 236 else 2, str(int(subid)), ha="center", va="top", fontsize=10
                         , color="gray")
                ax3.scatter(smooth_mdyn[-1], 1.6, marker="^"
                         , color="gray")
            
            
            
            ax4.remove()
            ax3.legend()
            ax2.legend()
            ax1.legend(loc='lower center', bbox_to_anchor=(0.4, -0.6), fancybox=True, title="Local Group Satellites (McConnachie 2012)", ncols=2)
            ax5.legend(loc='lower center', bbox_to_anchor=(-1.6, -0.8), fancybox=True, title=f"CS4G Isolated Dwarfs (Sánchez-Alarcón+, 2025)", ncols=2)
            
        
            fig.suptitle(f"Evolution of Observational parameters of AGORA-{name.upper()} Dwarfs during infall.", fontsize=27, y=0.92)
            fig.savefig(output_dir + f"/{name}.png", dpi=350)
    
            
            plt.close()
            
    
    
    else:
        k=-1
        for file, subtrees in files_subtrees.items():   
            k += 1
            table = pd.read_csv(file)
            grouped_table = table[table['Sub_tree_id'].isin(subtrees)].groupby("Sub_tree_id")
            name = plot_names[k]
            
            for (subid, candidate_df) in grouped_table:
                morph_mccon = mccon2
                
                trans_dict = {'dIrr/dSph': "p",
                              'dIrr': "*",
                              'dE/dSph': "h",
                              'cE': "D",
                              "Sc": "P",
                              'Irr': "s", 
                              '????': "x",
                              '(d)Irr?': "s",
                              'dSph' : "o",
                              'dSph?': "o"
                             }
                
                import matplotlib.gridspec as gridspec
                
                
                
                plt.rcParams['axes.linewidth'] = 1.1
                plt.rcParams['xtick.major.width'] = 1.1
                plt.rcParams['xtick.minor.width'] = 1.1
                plt.rcParams['ytick.major.width'] = 1.1
                plt.rcParams['ytick.minor.width'] = 1.1
                
                plt.rcParams['xtick.major.size'] = 7
                plt.rcParams['ytick.major.size'] = 7
                
                
                rh_lim = (1, 1E4)
                MV_lim = (2, -24)
                Mdyn_lim = (1E4, 9E10)
                sigma_lim = (1, 200)
                
                
                z = np.polyfit(np.log10(mccon['Mdyn'] * 1E6), np.log10(mccon['LV'] * 1E6 * 0.5), 1)
                p = np.poly1d(z)
                md = np.linspace(*Mdyn_lim, 1000)
                lhalf = 10 ** (z[0] * np.log10(md) + z[1])
                MV = 4.84 - 2.5 * np.log10(2*lhalf)
                
                
                fig = plt.figure(figsize=(1.5*1.8*9,1.8*9))
                    
                    
                gs = gridspec.GridSpec(2, 3, hspace=0, wspace=0)
                
                ax1 = fig.add_subplot(gs[0])  
                ax2 = fig.add_subplot(gs[1])  
                ax3 = fig.add_subplot(gs[4])  
                ax4 = fig.add_subplot(gs[3])  
                
                ax5 = fig.add_subplot(gs[2])  
                ax6 = fig.add_subplot(gs[5])  
                
                
                j = 0
                
                i = 0
                for key, value in trans_dict.items():
                    morph_mccon = mccon2[mccon2['MType'] == key]
                    
                    if key != "dSph?" and key != "(d)Irr?":
                        ax1.scatter([np.nan],[np.nan], marker=value, color="black", s=25, label=f"{key}") 
                
                        
                    if i==8:
                        a1 = ax1.scatter(morph_mccon['R2'][morph_mccon['SubG'] == "MW"], morph_mccon['VMag'][morph_mccon['SubG'] == "MW"], marker=value, color="red", s=25, label=f"Milky-Way group")
                        a2 = ax1.scatter(morph_mccon['R2'][morph_mccon['SubG'] == "M31"], morph_mccon['VMag'][morph_mccon['SubG'] == "M31"], marker=value, color="blue",s=25, label=f"Andromeda gruop")
                        a3 = ax1.scatter(morph_mccon['R2'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], morph_mccon['VMag'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], marker=value, color="green", s=25, label=f"Local group "+r"($D_{LG} \leq 1Mpc$)")
                        a4 = ax1.scatter(morph_mccon['R2'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], morph_mccon['VMag'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], marker=value, color="magenta", s=25, label=f"Nearby galaxies "+r"($D_{LG} > 1Mpc$)")
                    else:
                        ax1.scatter(morph_mccon['R2'][morph_mccon['SubG'] == "MW"], morph_mccon['VMag'][morph_mccon['SubG'] == "MW"], marker=value, color="red", s=25 )
                        ax1.scatter(morph_mccon['R2'][morph_mccon['SubG'] == "M31"], morph_mccon['VMag'][morph_mccon['SubG'] == "M31"], marker=value, color="blue",s=25 )
                        ax1.scatter(morph_mccon['R2'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], morph_mccon['VMag'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], marker=value, color="green", s=25 )
                        ax1.scatter(morph_mccon['R2'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], morph_mccon['VMag'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], marker=value, color="magenta", s=25 ) 
                
                
                    
                    ax1.set_xscale("log")
                    ax1.invert_yaxis()
                    ax1.set_xlim(*rh_lim)
                    ax1.set_ylim(*MV_lim)
                    ax1.set_xlabel(r"$r_h$ [pc]")
                    ax1.set_xticklabels([0, 1, 10, 100, 1000])
                    ax1.set_ylabel(r"$M_V$ [mag]" )
                    ax1.legend()
                
                    i += 1
                    
                    
                    
                    
                    y = morph_mccon['R2'] 
                    x = morph_mccon['Mdyn'] * 1E6 
                    ax3.scatter(x[morph_mccon['SubG'] == "MW"], y[morph_mccon['SubG'] == "MW"], marker=value, color="red", s=25, )
                    ax3.scatter(x[morph_mccon['SubG'] == "M31"], y[morph_mccon['SubG'] == "M31"], marker=value, color="blue",s=25, )
                    ax3.scatter(1E6 * morph_mccon['Mdyn'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], morph_mccon['R2'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], marker=value, color="green", s=25, )
                    ax3.scatter(1E6 * morph_mccon['Mdyn'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], morph_mccon['R2'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], marker=value, color="magenta", s=25, )
                    
                    ax3.set_ylabel(r"$r_h$ [pc]")
                    ax3.set_xlabel(r"$M_{dyn} (\leq r_h) \ \ [M_\odot]$")
                    ax3.loglog()
                    ax3.set_xlim(*Mdyn_lim)
                    ax3.set_ylim(ax1.get_xlim())
                    ax3.set_yticklabels(ax1.get_xticklabels())
                    
                    
                    
                    
                    
                    
                    y = morph_mccon['VMag']
                    x = 1E6 * morph_mccon['Mdyn']
                    ax2.scatter(x[morph_mccon['SubG'] == "MW"], y[morph_mccon['SubG'] == "MW"], marker=value, color="red", s=25, )
                    ax2.scatter(x[morph_mccon['SubG'] == "M31"], y[morph_mccon['SubG'] == "M31"], marker=value, color="blue",s=25, )
                    ax2.scatter(x[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], y[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], marker=value, color="green", s=25, )
                    ax2.scatter(x[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], y[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], marker=value, color="magenta", s=25, )
                    
                    
                     
                    
                    
                    ax2.set_xscale("log")
                    ax2.invert_yaxis()
                    ax2.set_ylim(ax1.get_ylim())
                    ax2.set_ylabel(None)
                    ax2.set_yticklabels([])
                    ax2.set_xticklabels([])
                    ax2.set_xlabel(None)
                    ax2.set_xlim(ax3.get_xlim())
                
                
                
                    x = morph_mccon['sigma*']
                    y = morph_mccon['R2']
                    ax6.scatter(x[morph_mccon['SubG'] == "MW"], y[morph_mccon['SubG'] == "MW"], marker=value, s=25, color="red")
                    ax6.scatter(x[morph_mccon['SubG'] == "M31"], y[morph_mccon['SubG'] == "M31"],  marker=value, s=25, color="blue")
                    ax6.scatter(x[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], y[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], marker=value, color="green", s=25, )
                    ax6.scatter(x[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], y[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], marker=value, color="magenta", s=25, )
                    
                    ax6.set_xscale("log")
                    ax6.set_yscale("log")
                    ax6.set_xlim(*sigma_lim)
                    ax6.set_xlabel(r"$\sigma_*$ [km/s]")
                    ax6.set_ylabel("")
                    ax6.set_yticklabels([])
                    ax6.set_ylim(ax3.get_ylim())
                
                
                
                
                
                    x = morph_mccon['sigma*']
                    y = morph_mccon['VMag']
                    ax5.scatter(x[morph_mccon['SubG'] == "MW"], y[morph_mccon['SubG'] == "MW"], marker=value, s=25, color="red")
                    ax5.scatter(x[morph_mccon['SubG'] == "M31"], y[morph_mccon['SubG'] == "M31"],  marker=value, s=25, color="blue")
                    ax5.scatter(x[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], y[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], marker=value, color="green", s=25, )
                    ax5.scatter(x[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], y[(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], marker=value, color="magenta", s=25, )    
                    
                    
                    ax5.set_xscale("log")
                    ax5.set_yscale("linear")
                    ax5.set_xlim(ax6.get_xlim())
                    ax5.set_xlabel("")
                    ax5.set_ylabel("")
                    ax5.set_yticklabels([])
                    ax5.set_xticklabels([])
                    ax5.set_ylim(ax1.get_ylim())
                
                
                legend_GC = ax1.scatter(globc["rh"], globc['MVt'], s=10, label="MW globular clusters")
                ax2.scatter(globc["Mdyn"], globc['MVt'], s=10)
                ax3.scatter(globc["Mdyn"], globc['rh'], s=10)
                ax6.scatter(globc["sigma0"], globc['rh'], s=10)
                ax5.scatter(globc["sigma0"], globc['MVt'], s=10)
    
                bf = ax2.plot(md, df['beta0'].mean() + df['beta1'].mean()*np.log10(md), lw=2, zorder=10, label="Best Fit")
                
                for beta0, beta1 in df[['beta0', 'beta1']].values[::1]:
                    ax2.plot(md, beta0 + beta1 * np.log10(md), lw=1, alpha=0.1, color="lightblue", zorder=-1)
                
                
                
                trans_dict_s4g = {
                    'E': "o",
                    'E-S0': "p",
                    'S0': "d",
                    'S0-a': "d",
                    "SABc": "s",
                    'SBa': "x", 
                    'SBb': "x",
                    'SBbc': "x",
                    'Sbc' : "P",
                    'Sc': "P",
                    'Sm': "P"
                }
                labels_s4g = {
                    'E': "E ",
                    'E-S0': "E-S0 ",
                    'S0': "S0 ",
                    "SABc": "SAB ",
                    'SBa': "SB ", 
                    'Sm': "S "
                }
                
                for key, value in trans_dict_s4g.items():
                    morph_s4g = s4g_sample[s4g_sample['type'] == key]
                
                    if key in ["E", "S0", "E-S0", "SABc", "SBa", "Sm"]:
                        ax5.scatter([np.nan],[np.nan], marker=value, color="tomato", s=25, label=f"{labels_s4g[key]}") 
                
                    b1 = ax1.scatter(morph_s4g["R2"], morph_s4g["VMag"], marker=value, color="tomato", s=25)
                    b1 = ax3.scatter(morph_s4g["Mdyn"], morph_s4g["R2"], marker=value, color="tomato", s=25)
                    b1 = ax2.scatter(morph_s4g["Mdyn"], morph_s4g["VMag"], marker=value, color="tomato", s=25)
                    b1 = ax6.scatter(morph_s4g["vdis"], morph_s4g["R2"], marker=value, color="tomato", s=25)       
                    b1 = ax5.scatter(morph_s4g["vdis"], morph_s4g["VMag"], marker=value, color="tomato", s=25)
                    
                
                
                
                candidate_df = candidate_df.sort_values("Snapshot", ascending=True)
                n = len(candidate_df)
                
                print(f"{subid} df. shape: {n}")
                print(candidate_df)
                
                
                rrvir = candidate_df['R/Rvir'].values
                targets = [rrvir.min(), 0.5, 1, 2, 3, 4]
                indices = [np.abs(rrvir - target).argmin() for target in targets]
                rrvir_markers = ["*", "^", "^", "^", "^", "o"]
                
                MV_1 = Msun - 2.5 * np.log10(candidate_df['LV'])
                
                smooth = lambda x: np.convolve(x, np.ones(5) / 5, mode='same')
                
                if MV_1.values.shape != smooth(MV_1.values).shape: raise Exception(f"Smoothing went wrong.")
                
                smooth_MV = smooth(MV_1.values)
                smooth_rh = smooth(candidate_df['rh_stars_physical'].values)
                smooth_mdyn = smooth(candidate_df['Mdyn'].values)
                smoth_sigma = smooth(candidate_df['sigma*'].values)
                
                ax1.plot(smooth_rh * 1E3, smooth_MV, color="black")
                ax6.plot(smoth_sigma , smooth_rh * 1E3,  color="black")   
                ax5.plot(smoth_sigma , smooth_MV,  color="black")
                ax3.plot(smooth_mdyn, smooth_rh * 1E3,  color="black")
                ax2.plot(smooth_mdyn, smooth_MV, color="black")
                
                                    
                for index, marker in zip(indices, rrvir_markers): 
                    ax1.scatter(smooth_rh[index] * 1E3, smooth_MV[index], color="gray", marker=marker, s=15)
                    ax6.scatter(smoth_sigma[index] , smooth_rh[index] * 1E3,  color="gray", marker=marker, s=15)   
                    ax5.scatter(smoth_sigma[index] ,smooth_MV[index],  color="gray", marker=marker, s=15)
                    ax3.scatter(smooth_mdyn[index], smooth_rh[index] * 1E3,  color="gray", marker=marker, s=15)
                    ax2.scatter(smooth_mdyn[index], smooth_MV[index], color="gray", marker=marker, s=15)
                
                ax1.text(smooth_rh[-1]*1E3, 0 if subid != 5888 else -0.75, str(int(subid)), ha="center", va="top", fontsize=10
                         , color="gray")
                ax1.scatter(smooth_rh[-1]*1E3, -0.25, marker="^"
                         , color="gray")
                ax6.text(smoth_sigma[-1], 1.44 if subid != 236 else 2, str(int(subid)), ha="center", va="top", fontsize=10
                         , color="gray")
                ax6.scatter(smoth_sigma[-1], 1.6, marker="^"
                         , color="gray")
                ax3.text(smooth_mdyn[-1], 1.44 if subid != 236 else 2, str(int(subid)), ha="center", va="top", fontsize=10
                         , color="gray")
                ax3.scatter(smooth_mdyn[-1], 1.6, marker="^"
                         , color="gray")
                
                
                
                ax4.remove()
                ax3.legend()
                ax2.legend()
                ax1.legend(loc='lower center', bbox_to_anchor=(0.4, -0.6), fancybox=True, title="Local Group Satellites (McConnachie 2012)", ncols=2)
                ax5.legend(loc='lower center', bbox_to_anchor=(-1.6, -0.8), fancybox=True, title=f"CS4G Isolated Dwarfs (Sánchez-Alarcón+, 2025)", ncols=2)
                
                
                fig.suptitle(f"Evolution of Observational parameters of AGORA-{name.upper()} Dwarfs during infall.", fontsize=27, y=0.92)
                fig.savefig(output_dir + f"/{name}_{int(subid)}.png", dpi=350)
                
                plt.close()
    
    
    




print(f"\n"*5)






if pltstyle == "all" or pltstyle == "ML":


    if combine:
        k = -1
        for file, subtrees in files_subtrees.items():  
            k += 1
            name = plot_names[k]
    
            morph_mccon = mccon2
            
            trans_dict = {'dIrr/dSph': "p",
                          'dIrr': "*",
                          'dE/dSph': "h",
                          'cE': "D",
                          "Sc": "P",
                          'Irr': "s", 
                          '????': "x",
                          '(d)Irr?': "s",
                          'dSph' : "o",
                          'dSph?': "o"
                         }
            
            import matplotlib.gridspec as gridspec
            
            
            
            plt.rcParams['axes.linewidth'] = 1.1
            plt.rcParams['xtick.major.width'] = 1.1
            plt.rcParams['xtick.minor.width'] = 1.1
            plt.rcParams['ytick.major.width'] = 1.1
            plt.rcParams['ytick.minor.width'] = 1.1
            
            plt.rcParams['xtick.major.size'] = 7
            plt.rcParams['ytick.major.size'] = 7
            
            
            ML_lim = (0.1, 1E4)
            Lum_lim = (1E2, 1E11)
            Mdyn_lim = (1E4, 5E11)
            
            
    
            fig, axes = plt.subplots(ncols=3, figsize=(3.5 * 6, 6))
                
                
            ax1 = axes[0]
            ax2 = axes[1]
            ax3 = axes[2]
            
            
            i = 0
            for key, value in trans_dict.items():
                morph_mccon = mccon2[mccon2['MType'] == key]
    
                
                ax1.scatter(1E6 * morph_mccon['Mdyn'][morph_mccon['SubG'] == "MW"], 0.5 * mag_to_lum(morph_mccon['VMag'][morph_mccon['SubG'] == "MW"]), marker=value, color="red", s=25 )
                ax1.scatter(1E6 * morph_mccon['Mdyn'][morph_mccon['SubG'] == "M31"], 0.5 * mag_to_lum(morph_mccon['VMag'][morph_mccon['SubG'] == "M31"]), marker=value, color="blue",s=25 )
                ax1.scatter(1E6 * morph_mccon['Mdyn'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], 0.5 * mag_to_lum(morph_mccon['VMag'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)]), marker=value, color="green", s=25 )
                ax1.scatter(1E6 * morph_mccon['Mdyn'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], 0.5 * mag_to_lum(morph_mccon['VMag'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)]), marker=value, color="magenta", s=25 ) 
        
                ax1.set_xscale("log")
                ax1.set_xlim(*Mdyn_lim)
                ax1.set_ylim(*Lum_lim)
                ax1.set_xlabel(r"$M_{dyn} (\leq r_h) \ \ [M_\odot]$")
                ax1.set_ylabel(r"$L_{1/2,V} \ \ [L_\odot]$" )
            
                i += 1
                
                ax2.scatter(1E6 * morph_mccon['Mdyn'][morph_mccon['SubG'] == "MW"], morph_mccon['ML'][morph_mccon['SubG'] == "MW"], marker=value, color="red", s=25 )
                ax2.scatter(1E6 * morph_mccon['Mdyn'][morph_mccon['SubG'] == "M31"], morph_mccon['ML'][morph_mccon['SubG'] == "M31"], marker=value, color="blue",s=25 )
                ax2.scatter(1E6 * morph_mccon['Mdyn'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], morph_mccon['ML'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], marker=value, color="green", s=25 )
                ax2.scatter(1E6 * morph_mccon['Mdyn'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], morph_mccon['ML'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], marker=value, color="magenta", s=25 )             
                
                ax2.set_xscale("log")
                ax2.set_xlim(*Mdyn_lim)
                ax2.set_ylim(*ML_lim)
                ax2.set_ylabel(r"$\Upsilon_{dyn,V} (\leq r_h) \ \ [M_\odot]$")
                ax2.set_xlabel(r"$M_{dyn} (\leq r_h) \ \ [M_\odot]$")
                
    
                ax3.scatter(mag_to_lum(morph_mccon['VMag'][morph_mccon['SubG'] == "MW"]), morph_mccon['ML'][morph_mccon['SubG'] == "MW"], marker=value, color="red", s=25 )
                ax3.scatter(mag_to_lum(morph_mccon['VMag'][morph_mccon['SubG'] == "M31"]), morph_mccon['ML'][morph_mccon['SubG'] == "M31"], marker=value, color="blue",s=25 )
                ax3.scatter(mag_to_lum(morph_mccon['VMag'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)]), morph_mccon['ML'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], marker=value, color="green", s=25 )
                ax3.scatter(mag_to_lum(morph_mccon['VMag'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)]), morph_mccon['ML'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], marker=value, color="magenta", s=25 )             
                
                ax3.set_xscale("log")
                ax3.set_xlim(*Lum_lim)
                ax3.set_ylim(*ML_lim)
                ax3.set_ylabel(r"$\Upsilon_{dyn,V} (\leq r_h) \ \ [M_\odot]$")
                ax3.set_xlabel(r"$L_{V} \ \ [L_\odot]$" )
    
    
                
            globc["ML"] = globc["Mdyn"] / (0.5 * mag_to_lum(globc['MVt']))
            ax1.scatter(globc["Mdyn"], 0.5 * mag_to_lum(globc['MVt']), s=10)
            ax2.scatter(globc["Mdyn"], globc['ML'], s=10)
            ax3.scatter(mag_to_lum(globc['MVt']), globc["ML"], s=10)
    
            
    
            
            trans_dict_s4g = {
                'E': "o",
                'E-S0': "p",
                'S0': "d",
                'S0-a': "d",
                "SABc": "s",
                'SBa': "x", 
                'SBb': "x",
                'SBbc': "x",
                'Sbc' : "P",
                'Sc': "P",
                'Sm': "P"
            }
            labels_s4g = {
                'E': "E ",
                'E-S0': "E-S0 ",
                'S0': "S0 ",
                "SABc": "SAB ",
                'SBa': "SB ", 
                'Sm': "S "
            }
            s4g_sample["ML"] = s4g_sample["Mdyn"] / (0.5 * mag_to_lum(s4g_sample['VMag']))
    
            for key, value in trans_dict_s4g.items():
                morph_s4g = s4g_sample[s4g_sample['type'] == key]
    
                ax1.scatter(morph_s4g["Mdyn"], 0.5 * mag_to_lum(morph_s4g['VMag']), marker=value, color="tomato", s=25)
                ax2.scatter(morph_s4g["Mdyn"], morph_s4g['ML'], marker=value, color="tomato", s=25)
                ax3.scatter(mag_to_lum(morph_s4g['VMag']), morph_s4g["ML"], marker=value, color="tomato", s=25)
            
            
                                
            table = pd.read_csv(file)
            grouped_table = table[table['Sub_tree_id'].isin(subtrees)].groupby("Sub_tree_id")
            
            for (subid, candidate_df) in grouped_table:
                candidate_df = candidate_df.sort_values("Snapshot", ascending=True)
    
                candidate_df["ML"] = candidate_df["Mdyn"] / (0.5*candidate_df["LV"])
                n = len(candidate_df)
                
                print(f"{subid} df. shape: {n}")
                print(candidate_df)
                
                
                rrvir = candidate_df['R/Rvir'].values
                targets = [rrvir.min(), 0.5, 1, 2, 3, 4]
                indices = [np.abs(rrvir - target).argmin() for target in targets]
                rrvir_markers = ["*", "^", "^", "^", "^", "o"]
                
                MV_1 = Msun - 2.5 * np.log10(candidate_df['LV'])
                
                smooth = lambda x: np.convolve(x, np.ones(5) / 5, mode='same')
                
                if MV_1.values.shape != smooth(MV_1.values).shape: raise Exception(f"Smoothing went wrong.")
    
    
                smooth_L = smooth(candidate_df["LV"].values)
                smooth_Lh = smooth(0.5*candidate_df["LV"].values)
                smooth_ML = smooth(candidate_df['ML'].values)
                smooth_mdyn = smooth(candidate_df['Mdyn'].values)
                
                ax1.plot(smooth_mdyn, smooth_Lh, color="black")
                ax2.plot(smooth_mdyn , smooth_ML,  color="black")   
                ax3.plot(smooth_L , smooth_ML,  color="black")
                

                for index, marker in zip(indices, rrvir_markers): 
                    ax1.scatter(smooth_mdyn[index], smooth_Lh[index], color="gray", marker=marker, s=15)
                    ax2.scatter(smooth_mdyn[index], smooth_ML[index],  color="gray", marker=marker, s=15)   
                    ax3.scatter(smooth_L[index] ,smooth_ML[index],  color="gray", marker=marker, s=15)
                                    
                
                ax1.text(smooth_mdyn[-1], 1E3 if subid != 5888 else -0.75, str(int(subid)), ha="center", va="top", fontsize=10
                         , color="gray")
                ax1.scatter(smooth_mdyn[-1], 5E2, marker="^"
                         , color="gray")
                ax2.text(smooth_mdyn[-1], 1.44 if subid != 236 else 2, str(int(subid)), ha="center", va="top", fontsize=10
                         , color="gray")
                ax2.scatter(smooth_mdyn[-1], 1.6, marker="^"
                         , color="gray")
                ax3.text(smooth_L[-1], 1.44 if subid != 236 else 2, str(int(subid)), ha="center", va="top", fontsize=10
                         , color="gray")
                ax3.scatter(smooth_L[-1], 1.6, marker="^"
                         , color="gray")
            
            
            
        
            #fig.suptitle(f"Evolution of Observational parameters of AGORA-{name.upper()} Dwarfs during infall.", fontsize=27, y=0.92)
            fig.savefig(output_dir + f"/{name}_ML.png", dpi=350)
    
            
            plt.close()
    
    
    else:
        k = -1
        for file, subtrees in files_subtrees.items():   
            k += 1
            table = pd.read_csv(file)
            grouped_table = table[table['Sub_tree_id'].isin(subtrees)].groupby("Sub_tree_id")
            name = plot_names[k]
            
            for (subid, candidate_df) in grouped_table:
                morph_mccon = mccon2
                
                trans_dict = {'dIrr/dSph': "p",
                              'dIrr': "*",
                              'dE/dSph': "h",
                              'cE': "D",
                              "Sc": "P",
                              'Irr': "s", 
                              '????': "x",
                              '(d)Irr?': "s",
                              'dSph' : "o",
                              'dSph?': "o"
                             }
                
                import matplotlib.gridspec as gridspec
                
                
                
                plt.rcParams['axes.linewidth'] = 1.1
                plt.rcParams['xtick.major.width'] = 1.1
                plt.rcParams['xtick.minor.width'] = 1.1
                plt.rcParams['ytick.major.width'] = 1.1
                plt.rcParams['ytick.minor.width'] = 1.1
                
                plt.rcParams['xtick.major.size'] = 7
                plt.rcParams['ytick.major.size'] = 7
                
                
                ML_lim = (0.1, 7E3)
                Lum_lim = (1E2, 1E11)
                Mdyn_lim = (1E3, 2E11)
                
                
        
                fig, axes = plt.subplots(ncols=3, figsize=(3.5 * 6, 6))
                    
                    
                ax1 = axes[0]
                ax2 = axes[1]
                ax3 = axes[2]
                
                
                i = 0
                for key, value in trans_dict.items():
                    morph_mccon = mccon2[mccon2['MType'] == key]
        
                    
                    ax1.scatter(1E6 * morph_mccon['Mdyn'][morph_mccon['SubG'] == "MW"], 0.5 * mag_to_lum(morph_mccon['VMag'][morph_mccon['SubG'] == "MW"]), marker=value, color="red", s=25 )
                    ax1.scatter(1E6 * morph_mccon['Mdyn'][morph_mccon['SubG'] == "M31"], 0.5 * mag_to_lum(morph_mccon['VMag'][morph_mccon['SubG'] == "M31"]), marker=value, color="blue",s=25 )
                    ax1.scatter(1E6 * morph_mccon['Mdyn'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], 0.5 * mag_to_lum(morph_mccon['VMag'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)]), marker=value, color="green", s=25 )
                    ax1.scatter(1E6 * morph_mccon['Mdyn'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], 0.5 * mag_to_lum(morph_mccon['VMag'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)]), marker=value, color="magenta", s=25 ) 
            
                    ax1.set_xscale("log")
                    ax1.set_yscale("log")
                    ax1.set_xlim(*Mdyn_lim)
                    ax1.set_ylim(*Lum_lim)
                    ax1.set_xlabel(r"$M_{dyn} (\leq r_h) \ \ [M_\odot]$")
                    ax1.set_ylabel(r"$L_{1/2,V} \ \ [L_\odot]$" )
                
                    i += 1
                    
                    ax2.scatter(1E6 * morph_mccon['Mdyn'][morph_mccon['SubG'] == "MW"], morph_mccon['ML'][morph_mccon['SubG'] == "MW"], marker=value, color="red", s=25 )
                    ax2.scatter(1E6 * morph_mccon['Mdyn'][morph_mccon['SubG'] == "M31"], morph_mccon['ML'][morph_mccon['SubG'] == "M31"], marker=value, color="blue",s=25 )
                    ax2.scatter(1E6 * morph_mccon['Mdyn'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], morph_mccon['ML'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], marker=value, color="green", s=25 )
                    ax2.scatter(1E6 * morph_mccon['Mdyn'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], morph_mccon['ML'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], marker=value, color="magenta", s=25 )             
                    
                    ax2.set_xscale("log")
                    ax2.set_yscale("log")
                    ax2.set_xlim(*Mdyn_lim)
                    ax2.set_ylim(*ML_lim)
                    ax2.set_ylabel(r"$\Upsilon_{dyn,V} (\leq r_h) \ \ [M_\odot]$")
                    ax2.set_xlabel(r"$M_{dyn} (\leq r_h) \ \ [M_\odot]$")
                    
        
                    ax3.scatter(mag_to_lum(morph_mccon['VMag'][morph_mccon['SubG'] == "MW"]), morph_mccon['ML'][morph_mccon['SubG'] == "MW"], marker=value, color="red", s=25 )
                    ax3.scatter(mag_to_lum(morph_mccon['VMag'][morph_mccon['SubG'] == "M31"]), morph_mccon['ML'][morph_mccon['SubG'] == "M31"], marker=value, color="blue",s=25 )
                    ax3.scatter(mag_to_lum(morph_mccon['VMag'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)]), morph_mccon['ML'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] <= 1000)], marker=value, color="green", s=25 )
                    ax3.scatter(mag_to_lum(morph_mccon['VMag'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)]), morph_mccon['ML'][(morph_mccon['SubG'] == "Rest") & (morph_mccon['D(LG)'] > 1000)], marker=value, color="magenta", s=25 )             
                    
                    ax3.set_xscale("log")
                    ax3.set_yscale("log")
                    ax3.set_xlim(Lum_lim[0], 3*Lum_lim[1])
                    ax3.set_ylim(*ML_lim)
                    ax3.set_ylabel(r"$\Upsilon_{dyn,V} (\leq r_h) \ \ [M_\odot]$")
                    ax3.set_xlabel(r"$L_{V} \ \ [L_\odot]$" )
        
        
                    
                globc["ML"] = globc["Mdyn"] / (0.5 * mag_to_lum(globc['MVt']))
                ax1.scatter(globc["Mdyn"], 0.5 * mag_to_lum(globc['MVt']), s=10)
                ax2.scatter(globc["Mdyn"], globc['ML'], s=10)
                ax3.scatter(mag_to_lum(globc['MVt']), globc["ML"], s=10)
        
                
        
                
                trans_dict_s4g = {
                    'E': "o",
                    'E-S0': "p",
                    'S0': "d",
                    'S0-a': "d",
                    "SABc": "s",
                    'SBa': "x", 
                    'SBb': "x",
                    'SBbc': "x",
                    'Sbc' : "P",
                    'Sc': "P",
                    'Sm': "P"
                }
                labels_s4g = {
                    'E': "E ",
                    'E-S0': "E-S0 ",
                    'S0': "S0 ",
                    "SABc": "SAB ",
                    'SBa': "SB ", 
                    'Sm': "S "
                }
                s4g_sample["ML"] = s4g_sample["Mdyn"] / (0.5 * mag_to_lum(s4g_sample['VMag']))
        
                for key, value in trans_dict_s4g.items():
                    morph_s4g = s4g_sample[s4g_sample['type'] == key]
        
                    ax1.scatter(morph_s4g["Mdyn"], 0.5 * mag_to_lum(morph_s4g['VMag']), marker=value, color="tomato", s=25)
                    ax2.scatter(morph_s4g["Mdyn"], morph_s4g['ML'], marker=value, color="tomato", s=25)
                    ax3.scatter(mag_to_lum(morph_s4g['VMag']), morph_s4g["ML"], marker=value, color="tomato", s=25)
                
                
                                    
    
                candidate_df = candidate_df.sort_values("Snapshot", ascending=True)
    
                candidate_df["ML"] = candidate_df["Mdyn"] / (0.5*candidate_df["LV"])
                n = len(candidate_df)
                
                print(f"{subid} df. shape: {n}")
                print(candidate_df)
                
                
                rrvir = candidate_df['R/Rvir'].values
                targets = [rrvir.min(), 0.5, 1, 2, 3, 4]
                indices = [np.abs(rrvir - target).argmin() for target in targets]
                rrvir_markers = ["*", "^", "^", "^", "^", "o"]
                
                MV_1 = Msun - 2.5 * np.log10(candidate_df['LV'])
                
                smooth = lambda x: np.convolve(x, np.ones(5) / 5, mode='same')
                
                if MV_1.values.shape != smooth(MV_1.values).shape: raise Exception(f"Smoothing went wrong.")
    
    
                smooth_L = smooth(candidate_df["LV"].values)
                smooth_Lh = smooth(0.5*candidate_df["LV"].values)
                smooth_ML = smooth(candidate_df['ML'].values)
                smooth_mdyn = smooth(candidate_df['Mdyn'].values)
                
                ax1.plot(smooth_mdyn, smooth_Lh, color="black")
                ax2.plot(smooth_mdyn, smooth_ML,  color="black")   
                ax3.plot(smooth_L, smooth_ML,  color="black")
                
                
                for index, marker in zip(indices, rrvir_markers): 
                    ax1.scatter(smooth_mdyn[index], smooth_Lh[index], color="gray", marker=marker, s=15)
                    ax2.scatter(smooth_mdyn[index], smooth_ML[index],  color="gray", marker=marker, s=15)   
                    ax3.scatter(smooth_L[index] ,smooth_ML[index],  color="gray", marker=marker, s=15)
                
                ax1.text(smooth_mdyn[-1], 4.5E2 if subid != 5888 else -0.75, str(int(subid)), ha="center", va="top", fontsize=10
                         , color="gray")
                ax1.scatter(smooth_mdyn[-1], 2.1E2, marker="^"
                         , color="gray")
                ax2.text(smooth_mdyn[-1], 0.22, str(int(subid)), ha="center", va="top", fontsize=10
                         , color="gray")
                ax2.scatter(smooth_mdyn[-1], 0.15, marker="^"
                         , color="gray")
                ax3.text(smooth_L[-1], 0.22 , str(int(subid)), ha="center", va="top", fontsize=10
                         , color="gray")
                ax3.scatter(smooth_L[-1], 0.15, marker="^"
                         , color="gray")
            
                
                
    
            
                #fig.suptitle(f"Evolution of Observational parameters of AGORA-{name.upper()} Dwarfs during infall.", fontsize=27, y=0.92)
                fig.savefig(output_dir + f"/{name}_{int(subid)}_ML.png", dpi=350)
        
                
                plt.close()
    
    
    
    
    

















