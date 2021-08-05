from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bentdna.shapefourier import ShapeAgent

class LpWindows:
    n_total_frames = 50000
    bp_id_first = 3
    bp_id_last = 17
    n_begin = 1
    n_end = 9

    def __init__(self, workfolder, host, n_frames_per_window):
        self.workfolder = workfolder
        self.host = host
        self.n_frames_per_window = n_frames_per_window

        self.host_folder = path.join(workfolder, host)
        self.n_list = list(range(self.n_begin, self.n_end+1))
        self.n_modes = len(self.n_list)
        self.an_folder = path.join(self.host_folder, 'an_folder')

        self.split_frame_list = self.ini_split_frame_list()
        self.n_windows = len(self.split_frame_list)
        self.split_df_list = None

        self.df_an = None

        self.f_lp_store_array = path.join(self.an_folder, 'lp_store_array.npy')
        self.lp_store_array = None

    def read_df_an(self):
        f_in = path.join(self.an_folder, f'an_{self.n_begin-1}_{self.n_end}_bpfirst{self.bp_id_first}_bplast{self.bp_id_last}.csv')
        self.df_an = pd.read_csv(f_in)
        print(f'Read df_an from {f_in}')

    def ini_split_frame_list(self):
        middle_interval = self.n_frames_per_window / 2
        split_frame_list = list()

        frame_id_1 = 0
        frame_id_2 = frame_id_1 + self.n_frames_per_window - 1

        execute_loop = True
        while execute_loop:
            if frame_id_2 > (self.n_total_frames + 1):
                execute_loop = False
                break
            split_frame_list.append((int(frame_id_1), int(frame_id_2)))
            frame_id_1 += middle_interval
            frame_id_2 = frame_id_1 + self.n_frames_per_window - 1
        return split_frame_list
        
    def get_split_frame_list(self):
        return self.split_frame_list

    def set_split_df_list(self):
        split_df_list = list()
        for frame_id_1, frame_id_2 in self.split_frame_list:
            split_df_list.append(self.df_an.iloc[frame_id_1:frame_id_2+1])
        self.split_df_list = split_df_list

    def get_split_df_list(self):
        return self.split_df_list
        
    def get_appr_L(self):
        return 0.34 * (self.bp_id_last - self.bp_id_first) # Unit: nm

    def get_Lp_array(self, wn_id):
        appr_L = self.get_appr_L()
        Lp_list = list()
        df = self.split_df_list[wn_id]
        for n in self.n_list:
            var_an = df[str(n)].var()
            Lp = np.square(appr_L) / (np.square(n) * np.square(np.pi) * var_an)
            Lp_list.append(Lp)
        return np.array(Lp_list)
        
    def set_lp_store_array(self):
        lp_store_array = np.zeros((self.n_windows, self.n_modes))
        for wn_id in range(self.n_windows):
            print(f'Start to process Window-{wn_id}')
            lp_store_array[wn_id,:] = self.get_Lp_array(wn_id)
        self.lp_store_array = lp_store_array
    
    def save_lp_store_array(self):
        np.save(self.f_lp_store_array, self.lp_store_array)
        print(f'Save Lp of all windows into {self.f_lp_store_array}')

    def read_lp_store_array(self):
        self.lp_store_array = np.load(self.f_lp_store_array)
        print(f'Read Lp from {self.f_lp_store_array}')

    def get_lp_store_array(self):
        return self.lp_store_array

    def get_nlist_mean_std_array_for_windows(self, n_begin, n_end):
        n_list = list(range(n_begin, n_end+1))
        n_modes = len(n_list)
        mean_array = np.zeros(n_modes)
        std_array = np.zeros(n_modes)
        for idx, mode_id in enumerate(n_list):
            mean_array[idx] = self.lp_store_array[:, mode_id-1].mean()
            std_array[idx] = self.lp_store_array[:, mode_id-1].std()
        return n_list, mean_array, std_array


class LpSixPlots:
    hosts = ['a_tract_21mer', 'g_tract_21mer']
    d_colors = {'a_tract_21mer': 'blue', 'atat_21mer': 'cyan',
                'g_tract_21mer': 'red', 'gcgc_21mer': 'magenta'}
    abbr_hosts = {'a_tract_21mer': 'poly(dA:dT)', 'ctct_21mer': 'CTCT', 'gcgc_21mer': 'poly(GC)',
                  'g_tract_21mer': 'poly(dG:dC)', 'atat_21mer': 'poly(AT)', 'tgtg_21mer': 'TGTG'} 
    workfolder = '/home/yizaochen/codes/dna_rna/length_effect/find_helical_axis'
    n_begin = 2
    n_end = 6
    n_bp = 15

    def __init__(self, figsize):
        self.figsize = figsize
        self.lbfz = 12
        self.lgfz = 12
        self.ticksize = 10

    def plot_main(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.figsize)
        for host in self.hosts:
            nlist, Lplist = self.get_nlist_Lplist(host)
            ax.plot(nlist, Lplist, linestyle='solid', marker='o', linewidth=1, 
                    markersize=4, color=self.d_colors[host], label=self.abbr_hosts[host])
        #ax.axvline(3, color='grey', linestyle='--', alpha=0.3)
        ax.axhline(50, color='grey', linestyle='--', alpha=0.5, label='Experimental $L_p$')
        ax.set_ylabel(r'$L_p$ (nm)', fontsize=self.lbfz)
        ax.set_xlabel('Mode number, n', fontsize=self.lbfz)
        ax.legend(frameon=False, fontsize=self.lgfz, ncol=1)
        ax.set_xticks(nlist)
        ax.tick_params(axis='both', labelsize=self.ticksize)
        return fig, ax

    def plot_main_wavelength_version(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.figsize)
        for host in self.hosts:
            nlist, Lplist = self.get_nlist_Lplist(host)
            wavelength_list = self.get_wavelengthlist(nlist)
            ax.plot(wavelength_list, Lplist, linestyle='solid', marker='o', linewidth=1, 
                    markersize=4, color=self.d_colors[host], label=self.abbr_hosts[host])
        #ax.axvline(3, color='grey', linestyle='--', alpha=0.3)
        ax.axhline(50, color='grey', linestyle='--', alpha=0.5, label='Experimental $L_p$')
        ax.set_ylabel(r'$L_p$ (nm)', fontsize=self.lbfz)
        ax.set_xlabel('wavelength (nm)', fontsize=self.lbfz)
        ax.legend(frameon=False, fontsize=self.lgfz, ncol=1)
        ax.set_xticks(wavelength_list[::-1])
        #ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.tick_params(axis='both', labelsize=self.ticksize)
        return fig, ax

    def get_nlist_Lplist(self, host):
        s_agent = ShapeAgent(self.workfolder, host)
        df_an = s_agent.read_df_an(0, 9)
        L = s_agent.get_appr_L()
        n_list = list(range(self.n_begin, self.n_end+1))
        Lp_list = list()
        for n in n_list:
            var_an = df_an[str(n)].var()
            Lp = np.square(L) / (np.square(n) * np.square(np.pi) * var_an)
            Lp_list.append(Lp)
        return n_list, Lp_list

    def get_wavelengthlist(self, n_list):
        L_bar = (self.n_bp - 1) * 0.34
        return [ 2 * L_bar / n for n in n_list]