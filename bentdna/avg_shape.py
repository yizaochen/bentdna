from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bentdna.shapefourier import ShapeAgent

class STheta:
    @staticmethod
    def get_filter_df(df, frame_id, bp_id_first, bp_id_last):
        mask = (df['i'] == bp_id_first)
        df0 = df[mask]
        mask = (df0['Frame_ID'] == frame_id)
        df1 = df0[mask]
        mask = (df1['j'].between(bp_id_first+1, bp_id_last))
        df2 = df1[mask]
        return df2

    @staticmethod
    def get_s_list(df):
        s_list = np.zeros(df.shape[0])
        delta_s_list = df['|l_j|'].tolist()
        s_total = 0
        for i, delta_s in enumerate(delta_s_list):
            s_total += delta_s
            s_list[i] = s_total
        return s_list

    @staticmethod
    def get_slist_thetalist(df, frame_id, bp_id_first, bp_id_last):
        df_filter = STheta.get_filter_df(df, frame_id, bp_id_first, bp_id_last)
        s_list = STheta.get_s_list(df_filter)
        theta_list = df_filter['theta'].tolist()
        s_list = [0] + list(s_list)
        theta_list = [0] + theta_list
        return s_list, theta_list

    @staticmethod
    def get_theta_array(df, frame_id, bp_id_first, bp_id_last):
        df_filter = STheta.get_filter_df(df, frame_id, bp_id_first, bp_id_last)
        theta_array = np.zeros(bp_id_last-bp_id_first+1)
        theta_array[1:] = df_filter['theta']
        return theta_array

class AvgShapeWindows:
    n_total_frames = 50000

    def __init__(self, workfolder, host, n_frames_per_window):
        self.workfolder = workfolder
        self.host = host
        self.n_frames_per_window = n_frames_per_window

        self.host_folder = path.join(workfolder, host)
        self.df_folder = path.join(self.host_folder, 'l_theta')
        self.n_bead = ShapeAgent.d_n_bp[host]
        self.f_l_modulus_theta = path.join(self.df_folder, f'l_modulus_theta_{self.n_bead}_beads.csv')
        self.df_l_modulus_theta = None

        self.split_frame_list = self.ini_split_frame_list()
        self.n_windows = len(self.split_frame_list)

        self.split_df_list = None

        self.f_shape_store_array = path.join(self.df_folder, 'shape_store_array.npy')
        self.shape_store_array = None
        
    def read_l_modulus_theta(self):
        if path.exists(self.f_l_modulus_theta):
            self.df_l_modulus_theta =  pd.read_csv(self.f_l_modulus_theta)
            print(f'Read {self.f_l_modulus_theta}')
        else:
            print(f'{self.f_l_modulus_theta} not exist!')

    def ini_split_frame_list(self):
        middle_interval = self.n_frames_per_window / 2
        split_frame_list = list()

        frame_id_1 = 1
        frame_id_2 = frame_id_1 + self.n_frames_per_window

        execute_loop = True
        while execute_loop:
            if frame_id_2 > (self.n_total_frames + 1):
                execute_loop = False
                break
            split_frame_list.append((int(frame_id_1), int(frame_id_2)))
            frame_id_1 += middle_interval
            frame_id_2 = frame_id_1 + self.n_frames_per_window
        return split_frame_list

    def get_split_frame_list(self):
        return self.split_frame_list

    def set_split_df_list(self):
        split_df_list = list()
        for frame_id_1, frame_id_2 in self.split_frame_list:
            mask = (self.df_l_modulus_theta['Frame_ID'] >= frame_id_1) & (self.df_l_modulus_theta['Frame_ID'] < frame_id_2)
            split_df_list.append(self.df_l_modulus_theta[mask])
        self.split_df_list = split_df_list

    def get_split_df_list(self):
        return self.split_df_list

    def get_shape_mean_array_by_wnid(self, wn_id):
        bp_id_first = 3
        bp_id_last = 17
        n_bp = self.n_bead-6
        start_frame, end_frame = self.split_frame_list[wn_id]

        temp_container = np.zeros((self.n_frames_per_window, n_bp))
        shape_mean_array = np.zeros(n_bp)

        df_sele = self.split_df_list[wn_id]
        frame_id_list = list(range(start_frame, end_frame))
        for idx, frame_id in enumerate(frame_id_list):
            temp_container[idx,:] = STheta.get_theta_array(df_sele, frame_id, bp_id_first, bp_id_last)

        for bp_id in range(n_bp):
            shape_mean_array[bp_id] = temp_container[:, bp_id].mean()
        return shape_mean_array

    def set_shape_store_array(self):
        n_bp = self.n_bead-6
        shape_store_array = np.zeros((self.n_windows, n_bp))
        for wn_id in range(self.n_windows):
            print(f'Start to process Window-{wn_id}')
            shape_store_array[wn_id,:] = self.get_shape_mean_array_by_wnid(wn_id)
        self.shape_store_array = shape_store_array
    
    def save_shape_store_array(self):
        np.save(self.f_shape_store_array, self.shape_store_array)
        print(f'Save shapes of all windows into {self.f_shape_store_array}')

    def read_shape_store_array(self):
        self.shape_store_array = np.load(self.f_shape_store_array)
        print(f'Read shape_store_array from {self.f_shape_store_array}')

    def get_shape_store_array(self):
        return self.shape_store_array

    def get_mean_std_array_for_windows(self):
        n_bp = self.n_bead-6
        mean_array = np.zeros(n_bp)
        std_array = np.zeros(n_bp)
        for bp_id in range(n_bp):
            mean_array[bp_id] = self.shape_store_array[:, bp_id].mean()
            std_array[bp_id] = self.shape_store_array[:, bp_id].std()
        return mean_array, std_array

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