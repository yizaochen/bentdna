from os import path, rename
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from bentdna.miscell import check_dir_exist_and_make

class ShapeAgent:
    d_n_bp = {
        'atat_21mer': 21, 'g_tract_21mer': 21, 'a_tract_21mer': 21,
        'yizao_model': 24, 'pnas_16mer': 16, 'gcgc_21mer': 21, 
        'tat_21mer': 21, 'tat_1_21mer': 21, 'tat_2_21mer': 21, 'tat_3_21mer': 21,
        'ctct_21mer': 21, 'tgtg_21mer': 21, '500mm': 16,
        'only_cation': 16, 'mgcl2_150mm': 16 }
        
    start_end = {
        'atat_21mer': (3, 17), 'g_tract_21mer': (3, 17), 
        'a_tract_21mer': (3, 17), 'yizao_model': (3, 20), 
        'pnas_16mer': (3, 12), 'gcgc_21mer': (3, 17),
        'ctct_21mer': (3, 17), 'tgtg_21mer': (3, 17),
        'tat_21mer': (3, 17), 'tat_1_21mer': (3, 17), 'tat_2_21mer': (3, 17), 'tat_3_21mer': (3, 17),
        '500mm': (3, 12), 'only_cation': (3, 12), 'mgcl2_150mm': (3, 12)}

    def __init__(self, workfolder, host, bp_id_first=None, bp_id_last=None):
        self.host = host
        self.rootfolder = path.join(workfolder, host)
        self.df_folder = path.join(self.rootfolder, 'l_theta')
        self.an_folder = path.join(self.rootfolder, 'an_folder')
        self.n_bead = self.d_n_bp[host]
        self.bp_id_first = self.__set_bp_id_first(bp_id_first)
        self.bp_id_last = self.__set_bp_id_last(bp_id_last)
        self.df_name = path.join(self.df_folder, f'l_modulus_theta_{self.n_bead}_beads.csv')
        self.df = None

        self.__check_and_make_folders()

    def get_appr_L(self):
        # Unit: nm
        return 0.34 * (self.bp_id_last - self.bp_id_first)
        
    def read_l_modulus_theta(self):
        self.df =  pd.read_csv(self.df_name)

    def get_L_of_frameid(self, frame_id):
        df_filter = self.get_filter_df(frame_id)
        L = self.__get_L(df_filter) # unit: angstrom
        return L

    def get_smid_and_interpolation_theta(self, frame_id):
        df_filter = self.get_filter_df(frame_id)
        theta_list = [0]
        theta_list += self.__get_theta_list(df_filter)
        n_theta = len(theta_list)
        s_mid_list = self.__get_s_mid_list(df_filter)
        interpolation_list = list()
        for i in range(n_theta-1):
            theta_inter = (theta_list[i] + theta_list[i+1]) / 2
            interpolation_list.append(theta_inter)
        return s_mid_list, interpolation_list

    def make_df_an(self, n_begin, n_end, last_frame):
        columns = list(range(n_begin, n_end+1))
        d_result = self.__initialize_an_d_result(n_begin, n_end)

        for frame_id in range(1, last_frame+1):
            df_filter = self.get_filter_df(frame_id)
            for n in columns:
                d_result[n].append(self.get_an(n, df_filter))

        df_an = pd.DataFrame(d_result)
        df_an = df_an[columns]
        f_out = path.join(self.an_folder, f'an_{n_begin}_{n_end}_bpfirst{self.bp_id_first}_bplast{self.bp_id_last}.csv')
        df_an.to_csv(f_out, index=False)
        return df_an

    def read_df_an(self, n_begin, n_end):
        f_in = path.join(self.an_folder, f'an_{n_begin}_{n_end}_bpfirst{self.bp_id_first}_bplast{self.bp_id_last}.csv')
        df_an = pd.read_csv(f_in)
        return df_an

    def get_filter_df(self, frame_id):
        mask = (self.df['i'] == self.bp_id_first)
        df0 = self.df[mask]
        mask = (df0['Frame_ID'] == frame_id)
        df1 = df0[mask]
        mask = (df1['j'].between(self.bp_id_first+1, self.bp_id_last))
        df2 = df1[mask]
        return df2

    def get_an(self, n, df_filter):
        L = self.__get_L(df_filter) # unit: angstrom
        L_nm = L / 10 # unit: nm
        delta_s_list = self.__get_delta_s_list(df_filter)
        theta_list = self.__get_theta_list(df_filter)
        s_mid_list = self.__get_s_mid_list(df_filter)
        scale_factor = np.sqrt(2/L_nm)
        summation = 0
        for delta_s, theta, s_mid in zip(delta_s_list, theta_list, s_mid_list):
            in_cos_term = n * np.pi / L
            cos_term = np.cos(in_cos_term * s_mid)
            summation += delta_s * 0.1 * theta * cos_term
        return scale_factor * summation

    def get_an_simplified(self, L, scale_factor, n, df_filter):
        delta_s_list = self.__get_delta_s_list(df_filter)
        theta_list = self.__get_theta_list(df_filter)
        s_mid_list = self.__get_s_mid_list(df_filter)
        summation = 0
        for delta_s, theta, s_mid in zip(delta_s_list, theta_list, s_mid_list):
            in_cos_term = n * np.pi / L
            cos_term = np.cos(in_cos_term * s_mid)
            summation += delta_s * 0.1 * theta * cos_term
        return scale_factor * summation

    def get_mode_shape_list(self, n, df_filter):
        L = self.__get_L(df_filter) # unit: angstrom
        L_nm = L / 10 # unit: nm
        s_list = np.array(self.__get_s_list(df_filter))
        scale_factor = np.sqrt(2/L_nm)
        in_cos_term = (n * np.pi * s_list) / L
        an = self.get_an(n, df_filter)
        cos_list = an * scale_factor * np.cos(in_cos_term)
        return s_list, cos_list

    def get_cos_list(self, s_list, L, scale_factor, n, df_filter):
        in_cos_term = (n * np.pi * s_list) / L
        an = self.get_an_simplified(L, scale_factor, n, df_filter)
        cos_list = an * scale_factor * np.cos(in_cos_term)
        return cos_list

    def get_cos_list_an(self, s_list, L, scale_factor, n, df_filter):
        in_cos_term = (n * np.pi * s_list) / L
        an = self.get_an_simplified(L, scale_factor, n, df_filter)
        cos_list = an * scale_factor * np.cos(in_cos_term)
        return cos_list, an

    def get_slist_thetalist(self, frame_id):
        df_filter = self.get_filter_df(frame_id)
        s_list = self.__get_s_list(df_filter)
        theta_list = self.__get_theta_list(df_filter)
        s_list = [0] + list(s_list)
        theta_list = [0] + theta_list
        return s_list, theta_list

    def get_approximate_theta(self, frame_id, n_begin, n_end):
        df_filter = self.get_filter_df(frame_id)
        L = self.__get_L(df_filter) # unit: angstrom
        L_nm = L / 10 # unit: nm
        scale_factor = np.sqrt(2/L_nm)
        s_list = self.__get_s_list(df_filter)
        appr_theta_list = np.zeros(len(s_list))
        for n in range(n_begin, n_end+1):
            cos_list = self.get_cos_list(s_list, L, scale_factor, n, df_filter)
            if n == 0:
                appr_theta_list += cos_list / 2
            else:
                appr_theta_list += cos_list
        s_list = [0] + list(s_list)
        appr_theta_list = [0] + list(appr_theta_list)
        return s_list, appr_theta_list

    def get_approximate_theta_singlemode(self, frame_id, n_select):
        df_filter = self.get_filter_df(frame_id)
        L = self.__get_L(df_filter) # unit: angstrom
        L_nm = L / 10 # unit: nm
        scale_factor = np.sqrt(2/L_nm)
        s_list = self.__get_s_list(df_filter)
        appr_theta_list = np.zeros(len(s_list))
        for n in [0, n_select]:
            cos_list, an = self.get_cos_list_an(s_list, L, scale_factor, n, df_filter)
            if n == 0:
                appr_theta_list += cos_list / 2
            else:
                appr_theta_list += cos_list
        s_list = [0] + list(s_list)
        appr_theta_list = [0] + list(appr_theta_list)
        return s_list, appr_theta_list, an

    def __set_bp_id_first(self, bp_id_first):
        if bp_id_first is None:
            return self.start_end[self.host][0]
        else:
            return bp_id_first

    def __set_bp_id_last(self, bp_id_last):
        if bp_id_last is None:
            return self.start_end[self.host][1]
        else:
            return bp_id_last

    def __initialize_an_d_result(self, n_begin, n_end):
        d_result = dict()
        for n in range(n_begin, n_end+1):
            d_result[n] = list()
        return d_result

    def __get_L(self, df):
        return df['|l_j|'].sum()
        
    def __get_theta_list(self, df):
        return df['theta'].tolist()
        
    def __get_delta_s_list(self, df):
        return df['|l_j|'].tolist()
        
    def __get_s_mid_list(self, df):
        s_mid_list = np.zeros(df.shape[0])
        delta_s_list = df['|l_j|'].tolist()
        s_total = 0
        for i, delta_s in enumerate(delta_s_list):
            s_mid = s_total + delta_s/2
            s_total += delta_s
            s_mid_list[i] = s_mid
        return s_mid_list
        
    def __get_s_list(self, df):
        s_list = np.zeros(df.shape[0])
        delta_s_list = df['|l_j|'].tolist()
        s_total = 0
        for i, delta_s in enumerate(delta_s_list):
            s_total += delta_s
            s_list[i] = s_total
        return s_list

    def __check_and_make_folders(self):
        for folder in [self.an_folder]:
            check_dir_exist_and_make(folder)

class AvgShapeAgent(ShapeAgent):
    def __init__(self, workfolder, host, bp_id_first=None, bp_id_last=None):
        super().__init__(workfolder, host, bp_id_first=None, bp_id_last=None)
        self.df_name = path.join(self.df_folder, f'l_modulus_theta_{self.n_bead}_beads_avg_structure.csv')

class AvgShapeSixPlots:
    hosts = ['a_tract_21mer', 'atat_21mer', 'ctct_21mer',
             'g_tract_21mer', 'gcgc_21mer', 'tgtg_21mer', 'tat_21mer']
    d_colors = {'a_tract_21mer': 'blue', 'atat_21mer': 'orange', 'ctct_21mer': 'green',
                'g_tract_21mer': 'red', 'gcgc_21mer': 'magenta', 'tgtg_21mer': 'cyan', 
                'tat_21mer': 'purple', 'tat_1_21mer': 'magenta', 'tat_2_21mer': 'green',
                'tat_3_21mer': 'cyan'}
    abbr_hosts = {'a_tract_21mer': 'A-tract', 'ctct_21mer': 'CTCT', 'gcgc_21mer': 'GCGC',
                  'g_tract_21mer': 'G-tract', 'atat_21mer': 'ATAT', 'tgtg_21mer': 'TGTG', 
                  'tat_21mer': 'TAT-0', 'tat_1_21mer': 'TAT-1', 'tat_2_21mer': 'TAT-2', 'tat_3_21mer': 'TAT-3'} 
    workfolder = '/home/yizaochen/codes/dna_rna/length_effect/find_helical_axis'
    nrows = 2
    ncols = 3
    start_frame = 1
    end_frame = 10000
    n_bp = 15

    def __init__(self, figsize):
        self.figsize = figsize
        self.df_name = path.join(self.workfolder, 'average_theta0_six_systems.csv')
        self.lbfz = 12
        self.lgfz = 12
        self.ticksize = 10

    def get_theta0_r18_radian(self):
        df = self.read_dataframe()
        for host in self.hosts:
            value = df[host].iloc[14]
            print(f'{host}: {value:.3f}')

    def get_theta0_r18_degree(self):
        df = self.read_dataframe()
        for host in self.hosts:
            value = np.rad2deg(df[host].iloc[14])
            print(f'{host}: {value:.1f}')

    def get_theta0_r15_degree(self):
        df = self.read_dataframe()
        for host in self.hosts:
            value = np.rad2deg(df[host].iloc[11])
            print(f'{host}: {value:.1f}')

    def make_dataframe(self):
        d_result = dict()
        for host in self.hosts:
            temp_container = np.zeros((self.end_frame, self.n_bp))
            d_result[host] = np.zeros(self.n_bp)
            s_agent = ShapeAgent(self.workfolder, host)
            s_agent.read_l_modulus_theta()
            for frameid in range(self.start_frame, self.end_frame+1):
                data = s_agent.get_slist_thetalist(frameid)
                temp_container[frameid-1,:] = data[1]
            for bp_id in range(self.n_bp):
                d_result[host][bp_id] = temp_container[:, bp_id].mean()
        df = pd.DataFrame(d_result)
        df.to_csv(self.df_name)
        print(f'Write Dataframe to {self.df_name}')
        return df

    def read_dataframe(self):
        df = pd.read_csv(self.df_name)
        return df

    def make_dataframe_sele_host(self, sele_host):
        d_result = dict()
        for host in [sele_host]:
            temp_container = np.zeros((self.end_frame, self.n_bp))
            d_result[host] = np.zeros(self.n_bp)
            s_agent = ShapeAgent(self.workfolder, host)
            s_agent.read_l_modulus_theta()
            for frameid in range(self.start_frame, self.end_frame+1):
                data = s_agent.get_slist_thetalist(frameid)
                temp_container[frameid-1,:] = data[1]
            for bp_id in range(self.n_bp):
                d_result[host][bp_id] = temp_container[:, bp_id].mean()
        df = pd.DataFrame(d_result)
        return df

    def plot_main(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.figsize)
        df = self.read_dataframe()
        hosts = ['a_tract_21mer', 'atat_21mer', 'ctct_21mer',
                 'g_tract_21mer', 'gcgc_21mer', 'tgtg_21mer']
        for host in hosts:
            ylist = df[host]
            xlist = range(len(ylist))
            ax.plot(xlist, np.rad2deg(ylist), linestyle='--', marker='.', linewidth=1, 
                    markersize=4, label=self.abbr_hosts[host], color=self.d_colors[host])
        ylabel = self.get_ylabel()
        #ax.axvline(11, color='grey', alpha=0.6)
        ax.set_ylabel(ylabel, fontsize=self.lbfz)
        ax.legend(frameon=False, fontsize=self.lgfz)
        xticks, xticklabels = self.get_xticks_xticklabels()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.tick_params(axis='both', labelsize=self.ticksize)
        return fig, ax

    def plot_a_junction(self):
        groups = ['a_tract_21mer', 'tat_21mer', 'tat_1_21mer','tat_2_21mer', 'tat_3_21mer', 'atat_21mer']
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.figsize)
        df = self.read_dataframe()
        for host in groups:
            ylist = df[host]
            xlist = range(len(ylist))
            ax.plot(xlist, np.rad2deg(ylist), linestyle='--', marker='.', linewidth=1, 
                    markersize=4, label=self.abbr_hosts[host], color=self.d_colors[host])
        ylabel = self.get_ylabel()
        ax.axvline(11, color='grey', alpha=0.6)
        ax.set_ylabel(ylabel, fontsize=self.lbfz)
        ax.legend(frameon=False, fontsize=self.lgfz)
        xticks, xticklabels = self.get_xticks_xticklabels()
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.tick_params(axis='both', labelsize=self.ticksize)
        return fig, ax

    def plot_six_plots(self):
        fig, axes = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=self.figsize)
        d_axes = self.get_d_axes(axes)
        df = self.read_dataframe()
        for host in self.hosts:
            ax = d_axes[host]
            ylist = df[host]
            xlist = range(len(ylist))
            ax.plot(xlist, ylist, '--.', color='blue')
            ax.set_ylabel(r"$\theta(s)$ (degree)")
            #ax.set_title()
        return fig, d_axes

    def get_ylabel(self):
        return r'$\theta^{0}(\mathbf{r}_i)$  (degree)'

    def get_xticks_xticklabels(self):
        xticks = range(self.n_bp)
        xticklabels = list()
        start_idx = 4
        for bp_id in xticks:
            idx = start_idx + bp_id
            xticklabels.append(r'$\mathbf{r}_{' + f'{idx}' +r'}$')
        return xticks, xticklabels

    def get_d_axes(self, axes):
        d_axes = dict()
        idx_host = 0
        for row_id in range(self.nrows):
            for col_id in range(self.ncols):
                host = self.hosts[idx_host]
                d_axes[host] = axes[row_id, col_id]
                idx_host += 1
        return d_axes

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
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
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

class LpBarPlots:
    hosts = ['a_tract_21mer', 'atat_21mer', 'g_tract_21mer', 'gcgc_21mer']
    d_colors = {'a_tract_21mer': 'blue', 'atat_21mer': 'cyan', 'g_tract_21mer': 'red', 'gcgc_21mer': 'magenta'}
    abbr_hosts = {'a_tract_21mer': 'poly(dA:dT)', 'gcgc_21mer': 'poly(GC)',
                  'g_tract_21mer': 'poly(dG:dC)', 'atat_21mer': 'poly(AT)'} 
    workfolder = '/home/yizaochen/codes/dna_rna/length_effect/find_helical_axis'
    
    def __init__(self, figsize):
        self.figsize = figsize
        self.lbfz = 14
        self.lgfz = 12
        self.ticksize = 12

    def plot_main(self, width, color):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=self.figsize)
        xlist = range(len(self.hosts))
        Lplist = [self.get_Lp(host, 3)  for host in self.hosts]
        ax.bar(xlist, Lplist, width, color=color)
        ax.set_ylabel(r'$L_p$ (nm)', fontsize=self.lbfz)
        ax.set_xticks(xlist)
        ax.set_xticklabels([self.abbr_hosts[host] for host in self.hosts])
        ax.axhline(45, color='magenta', linestyle='--')
        ax.axhline(50, color='magenta', linestyle='--', label='single-molecule experiments\nfor various sequences')
        ax.tick_params(axis='both', labelsize=self.ticksize)
        ax.legend(frameon=False, fontsize=self.lgfz)
        return fig, ax

    def get_Lp(self, host, n_given):
        s_agent = ShapeAgent(self.workfolder, host)
        df_an = s_agent.read_df_an(0, 9)
        L = s_agent.get_appr_L()
        var_an = df_an[str(n_given)].var()
        return np.square(L) / (np.square(n_given) * np.square(np.pi) * var_an)

class DecomposeDraw:
    xlabel_fz = 14

    def __init__(self, workfolder, host):
        self.workfolder = workfolder
        self.host = host
        self.s_agent = ShapeAgent(workfolder, host)

        self.s_agent.read_l_modulus_theta()

    def plot_decompose_by_frame_id(self, frame_id, figsize):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize)
        self.ax_plot_original(axes[0,0], frame_id)
        xlim = axes[0,0].get_xlim()
        self.ax_plot_decompose_by_n(axes[0,1], frame_id, 2, xlim)
        self.ax_plot_decompose_by_n(axes[0,2], frame_id, 3, xlim)
        self.ax_plot_decompose_by_n(axes[1,0], frame_id, 4, xlim)
        self.ax_plot_decompose_by_n(axes[1,1], frame_id, 5, xlim)
        self.ax_plot_decompose_by_n(axes[1,2], frame_id, 6, xlim)
        return fig, axes

    def ax_plot_original(self, ax, frame_id):
        s_list, theta_list = self.s_agent.get_slist_thetalist(frame_id)
        ylist, yticklabels = self.get_ylist_yticklabels(s_list)
        ax.plot(np.rad2deg(theta_list), ylist, '-o', color='red')
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_yticks(ylist)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel(r"$\theta(\mathbf{r}_{i})$ (degree)", fontsize=self.xlabel_fz)
        title = f'ATAT, The {frame_id}' + r'$^{\mathrm{th}}$ Frame'
        ax.set_title(title)

    def ax_plot_decompose_by_n(self, ax, frame_id, n, xlim):
        s_list, appr_theta_list = self.s_agent.get_approximate_theta(frame_id, 0, n)
        ylist, yticklabels = self.get_ylist_yticklabels(s_list)
        ax.plot(np.rad2deg(appr_theta_list), ylist, '-o', color='blue')
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_yticks(ylist)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel(self.get_xlabel(n), fontsize=self.xlabel_fz)
        ax.set_xlim(xlim)
        title = f'$n={n}$'
        ax.set_title(title)

    def get_xlabel(self, n):
        return r'$\sqrt{\frac{2}{L}}a_{' + f'{n}' + r'}\cos{(\frac{' + f'{n}' + r'\pi s}{L})}$'

    def get_ylist_yticklabels(self, s_list):
        ylist = range(len(s_list))
        yticklabels = list()
        start_idx = 4
        for bp_id in ylist:
            idx = start_idx + bp_id
            yticklabels.append(r'$\mathbf{r}_{' + f'{idx}' +r'}$')
        return ylist, yticklabels

class HistogramAn:

    n_begin = 0
    n_end = 9
    hosts = ['a_tract_21mer', 'atat_21mer', 'ctct_21mer',
             'g_tract_21mer', 'gcgc_21mer', 'tgtg_21mer']
    d_colors = {'a_tract_21mer': 'blue', 'atat_21mer': 'orange', 'ctct_21mer': 'green',
                'g_tract_21mer': 'red', 'gcgc_21mer': 'magenta', 'tgtg_21mer': 'cyan'}
    abbr_hosts = {'a_tract_21mer': 'A-tract', 'ctct_21mer': 'CTCT', 'gcgc_21mer': 'GCGC',
                  'g_tract_21mer': 'G-tract', 'atat_21mer': 'ATAT', 'tgtg_21mer': 'TGTG'}
    n_rows = 2
    n_cols = 3

    def __init__(self, workfolder):
        self.workfolder = workfolder

        self.d_agents = self.get_d_agents()
        self.d_df = self.get_d_df()

    def plot_main(self, figsize):
        n = 3
        fig, axes = plt.subplots(nrows=self.n_rows, ncols=self.n_cols, figsize=figsize, sharex=True, sharey=True)
        d_axes = self.get_d_axes_by_host(axes)
        for host in self.hosts:
            ax = d_axes[host]
            data = self.d_df[host][str(n)]
            mean = np.mean(data)
            std = np.std(data)
            ax.hist(data, color=self.d_colors[host], density=True, bins=100, label=self.abbr_hosts[host])
            ax.axvline(mean, color='black', alpha=0.5)
            self.plot_assist_x(ax)
            self.plot_assist_y(ax)
            ax.set_title(self.get_title(host, mean, std), fontsize=10)
            if host in ['g_tract_21mer', 'gcgc_21mer', 'tgtg_21mer']:
                ax.set_xlabel(r'$a_3$')
            if host in ['a_tract_21mer', 'g_tract_21mer']:
                ax.set_ylabel('Probability')
        return fig, d_axes

    def get_title(self, host, mean, std):
        return f'{self.abbr_hosts[host]}\n' + r'$\mu=' + f'{mean:.3f}' + r'~~\sigma=' + f'{std:.3f}' + r'$'

    def plot_assist_x(self, ax):
        xvalues = np.arange(-0.4, 0.4, 0.1)
        for xvalue in xvalues:
            ax.axvline(xvalue, color='grey', alpha=0.1)

    def plot_assist_y(self, ax):
        yvalues = range(1, 8)
        for yvalue in yvalues:
            ax.axhline(yvalue, color='grey', alpha=0.1)

    def get_d_axes_by_host(self, axes):
        d_axes = dict()
        host_id = 0
        for row_id in range(self.n_rows):
            for col_id in range(self.n_cols):
                host = self.hosts[host_id]
                d_axes[host] = axes[row_id, col_id]
                host_id += 1
        return d_axes

    def get_d_agents(self):
        d_agents = dict()
        for host in self.hosts:
            d_agents[host] = ShapeAgent(self.workfolder, host)
        return d_agents

    def get_d_df(self):
        d_df = dict()
        for host in self.hosts:
            d_df[host] = self.d_agents[host].read_df_an(self.n_begin, self.n_end)
        return d_df
