from os import path
from itertools import combinations
import pickle
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import MDAnalysis as mda
from miscell import get_modulus_angle_between_two_vectors, check_dir_exist_and_make

findh_folder = '/home/yizaochen/codes/dna_rna/length_effect/find_helical_axis'


class LmodulusTheta:
    d_n_bp = {
        'atat_21mer': 21, 'g_tract_21mer': 21, 'a_tract_21mer': 21,
        'yizao_model': 24, 'pnas_16mer': 16, 'gcgc_21mer': 21,
        'ctct_21mer': 21, 'tgtg_21mer': 21, '500mm': 16,
        'only_cation': 16, 'mgcl2_150mm': 16 }

    def __init__(self, host):
        self.host = host
        self.rootfolder = path.join(findh_folder, host)
        self.output_folder = path.join(self.rootfolder, 'output')
        self.df_folder = path.join(self.rootfolder, 'l_theta')

        self.pdb_in = path.join(self.output_folder, 'haxis.0.pdb')
        self.dcd_in = path.join(self.output_folder, 'haxis.dcd')
        self.u = mda.Universe(self.pdb_in, self.dcd_in)

        self.n_bead = self.d_n_bp[host]
        self.n_bead_minus_1 = self.n_bead - 1

        self.columns = ['Frame_ID', 'i', 'j', '|l_i|', '|l_j|', 'theta']
        self.d_result = self.__get_d()
        self.df_name = path.join(self.df_folder, f'l_modulus_theta_{self.n_bead}_beads.csv')
        self.df = None

        self.__check_and_make_folders()

    def read_l_modulus_theta(self):
        self.df =  pd.read_csv(self.df_name)

    def make_l_modulus_theta(self):
        pair_list = self.__get_pair_list()
        for ts in self.u.trajectory:
            vectors = self.__get_vectors()
            for i, j in pair_list:
                self.__append_to_d_result(vectors, i, j, ts)
        self.df = self.__covert_d_to_df()
        self.df.to_csv(self.df_name, index=False)
        print(f'make {self.df_name}')

    def get_ensemble_average_l(self):
        l_result = list()
        for frame_id in range(len(self.u.trajectory)):
            self.u.trajectory[frame_id]
            vectors = self.__get_vectors()
            l_result += [np.linalg.norm(vector) for vector in vectors]
        l_result = np.array(l_result)
        return l_result.mean()

    def __check_and_make_folders(self):
        for folder in [self.df_folder]:
            check_dir_exist_and_make(folder)
        
    def __get_d(self):  
        d = dict()
        for key in self.columns:
            d[key] = list()
        return d

    def __get_pair_list(self):
        return list(combinations(range(self.n_bead_minus_1), 2))

    def __get_vectors(self):
        points = self.u.atoms.positions
        return [points[i + 1] - points[i] for i in range(self.n_bead_minus_1)]

    def __append_to_d_result(self, vectors, i, j, ts):
        vi_modulus, vj_modulus, theta = get_modulus_angle_between_two_vectors(vectors[i], vectors[j])
        self.d_result['Frame_ID'].append(ts.frame)
        self.d_result['i'].append(i)
        self.d_result['j'].append(j)
        self.d_result['|l_i|'].append(vi_modulus)
        self.d_result['|l_j|'].append(vj_modulus)
        self.d_result['theta'].append(theta)
    
    def __covert_d_to_df(self):
        df = pd.DataFrame(self.d_result)
        return df[self.columns]


class PersistenceLength:
    def __init__(self, df_l_theta, l_avg=None):
        self.df_l_theta = df_l_theta
        self.l_avg = self.__set_l_avg(l_avg) # Ensemble Average l, unit= angstrom

    def get_lp_by_i_j(self, i, j):
        mask = (self.df_l_theta['i'] == i) & (self.df_l_theta['j'] == j)
        df_temp = self.df_l_theta[mask]
        delta_theta_array = df_temp['theta'].values - df_temp['theta'].mean()
        cos_delta_theta = np.cos(delta_theta_array)
        return (-(j-i) * self.l_avg) / (2 * np.log(cos_delta_theta.mean()))

    def __set_l_avg(self, l_avg):
        if l_avg is None:
            return 3.4
        else:
            return l_avg


class LengthEffect:
    start_end = {
        'atat_21mer': (3, 15), 'g_tract_21mer': (3, 15), 
        'a_tract_21mer': (3, 15), 'yizao_model': (3, 18), 
        'pnas_16mer': (3, 10), 'gcgc_21mer': (3, 15),
        'ctct_21mer': (3, 15), 'tgtg_21mer': (3, 15), '500mm': (3, 10),
        'only_cation': (3, 10), 'mgcl2_150mm': (3, 10)}
        
    def __init__(self, host, l_result):
        self.host = host
        self.start_i = self.start_end[host][0]
        self.end_i = self.start_end[host][1]
        self.start_j = self.start_i + 1
        self.end_j = self.end_i + 1
        self.columns = ['i', 'j', 'j-i', '<delta theta^2>', '<cos delta theta>', 'ln<cos delta theta>', '<(theta/2)^2>','lp']
        self.l_result = l_result  # List contains 5 split data
        self.df_container = None
        self.d_plots = None
        self.output_folder = path.join(findh_folder, host, 'output')

    def set_df_container(self):
        self.df_container = list()
        for l_idx in range(5):
            d_result = self.__get_d_result_by_l_idx(l_idx)
            self.df_container.append(pd.DataFrame(d_result))

    def set_d_plots(self):
        start = 1
        end = self.end_j - self.start_i
        keys = ['lp', 'ln<cos delta theta>', '<delta theta^2>', '<(theta/2)^2>']
        self.d_plots = self.__initial_d_plots(keys, start, end)
        for key in keys:
            self.__append_to_d_plots(self.d_plots, key)

    def save_d_plots_to_pkl(self):
        fname = path.join(self.output_folder, f'lp_lncos_deltatheta.pkl')
        with open(fname, 'wb') as handle:
            pickle.dump(self.d_plots, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f'Write {fname}')

    def read_d_plots_from_pkl(self):
        fname = path.join(self.output_folder, f'lp_lncos_deltatheta.pkl')
        with open(fname, 'rb') as handle:
            self.d_plots = pickle.load(handle)
            print(f'Read {fname}')

    def __get_d_result_by_l_idx(self, l_idx):
        d_result = self.__get_d()
        for i in range(self.start_i, self.end_i+1):
            for j in range(self.start_j, self.end_j+1):
                if j <= i:
                    continue
                d_result = self.__append_to_d_result(d_result, l_idx, i, j)
        return d_result

    def __append_to_d_result(self, d_result, l_idx, i, j):
        var, mean_cos, log_mean_cos, lp, sq_theta_div_2_mean = get_pl_stat(self.l_result[l_idx], i, j)
        d_result['i'].append(i)
        d_result['j'].append(j)
        d_result['j-i'].append(j-i)
        d_result['<delta theta^2>'].append(var)
        d_result['<cos delta theta>'].append(mean_cos)
        d_result['ln<cos delta theta>'].append(log_mean_cos)
        d_result['lp'].append(lp)
        d_result['<(theta/2)^2>'].append(sq_theta_div_2_mean)
        return d_result

    def __append_to_d_plots(self, d_plots, key):
        for j_minus_i in d_plots['j-i']:
            temp_list = list()
            for l_idx in range(5):
                df_select = self.df_container[l_idx]
                mask = (df_select['j-i'] == j_minus_i)
                df_temp = df_select[mask]
                temp_list.append(df_temp[key].mean())
            temp_list = np.array(temp_list)
            d_plots[key]['mean'].append(temp_list.mean())
            d_plots[key]['std'].append(temp_list.std())
        return d_plots

    def __get_d(self):  
        d = dict()
        for key in self.columns:
            d[key] = list()
        return d

    def __initial_d_plots(self, keys, start, end):
        d = {'j-i': range(start, end+1)}
        for key in keys:
            d[key] = {'mean': list(), 'std': list()}
        return d


class LpThreePlots:
    lbfz = 14
    d_ylabel = {
        '<delta theta^2>': r'$\left< \delta \theta^2 \right>$', 
        'ln<cos delta theta>':r'$\ln \left< \cos \delta \theta \right>$',
        'lp': r'$L_p$ (nm)'}
    d_color = {'<delta theta^2>': 'g', 'ln<cos delta theta>': 'r', 'lp': 'b'}

    def __init__(self, host, d_plots, ax1, ax2, ax3):
        self.host = host
        self.d_plots = d_plots
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3

    def plot_main(self):
        self.__plot_ax(self.ax1, 'lp')
        self.__plot_ax(self.ax2, 'ln<cos delta theta>')
        self.__plot_ax(self.ax3, '<delta theta^2>')

        self.__set_title_host(self.ax1)
        self.__set_scientific_yticks(self.ax2)
        self.__set_scientific_yticks(self.ax3)
    
    def set_ylim_by_d(self, d):
        self.ax1.set_ylim(d['lp'])
        self.ax2.set_ylim(d['ln<cos delta theta>'])
        self.ax3.set_ylim(d['<delta theta^2>'])

    def __get_xlist_ymean_ystd(self, key):
        xlist = self.d_plots['j-i']
        y_mean = self.d_plots[key]['mean']
        y_std = self.d_plots[key]['std']
        return xlist, y_mean, y_std

    def __set_xy_labels(self, ax, key):
        ax.set_ylabel(self.d_ylabel[key], fontsize=self.lbfz)
        ax.set_xlabel('j-i', fontsize=self.lbfz)

    def __set_title_host(self, ax):
        ax.set_title(self.host, fontsize=self.lbfz)

    def __set_scientific_yticks(self, ax):
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

    def __plot_ax(self, ax, key):
        xlist, y_mean, y_std = self.__get_xlist_ymean_ystd(key)
        ax.errorbar(xlist, y_mean, yerr=y_std, linestyle='-', marker='.', color=self.d_color[key], ecolor='grey', capsize=6)
        self.__set_xy_labels(ax, key)
        ax.set_xticks(xlist)


class LpSinglePlots:
    lbfz = 14
    lgfz = 12
    ylabel = r'$L_p$ (nm)'
    colors = ['r', 'g', 'b', 'cyan', 'magenta', 'orange']

    def __init__(self, hosts, figsize):
        self.hosts = hosts
        self.figsize = figsize
    
    def plot_main(self):
        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=self.figsize)
        for idx, host in enumerate(self.hosts):
            d_plots = self.__read_d_plots_from_pkl(host)
            xlist, y_mean, y_std = self.__get_xlist_ymean_ystd(d_plots)
            ax.errorbar(xlist, y_mean, yerr=y_std, label=host, linestyle='-', marker='.', color=self.colors[idx], ecolor='grey', capsize=6)
            self.__set_xy_labels(ax)
        ax.set_xticks(xlist)
        ax.legend(fontsize=self.lgfz, frameon=False)
        return fig, ax

    def __read_d_plots_from_pkl(self, host):
        output_folder = path.join(findh_folder, host, 'output')
        fname = path.join(output_folder, 'lp_lncos_deltatheta.pkl')
        with open(fname, 'rb') as handle:
            d_plots = pickle.load(handle)
        return d_plots

    def __get_xlist_ymean_ystd(self, d_plots):
        key = 'lp'
        xlist = d_plots['j-i']
        y_mean = d_plots[key]['mean']
        y_std = d_plots[key]['std']
        return xlist, y_mean, y_std

    def __set_xy_labels(self, ax):
        ax.set_ylabel(self.ylabel, fontsize=self.lbfz)
        ax.set_xlabel('j-i', fontsize=self.lbfz)

class MultipleHostPlots:
    def __init__(self, hosts, figsize):
        self.hosts = hosts
        self.n_host = len(hosts)
        self.figsize = figsize
        self.d_max_min = self.__initial_d_max_min()

    def plot_main(self):
        fig, axes = plt.subplots(ncols=3, nrows=self.n_host, figsize=self.figsize)
        for row_id, host in enumerate(self.hosts):
            d_plots = self.__read_d_plots_from_pkl(host)
            ax1, ax2, ax3 = self.__get_ax123(row_id, axes)
            threeplot = LpThreePlots(host, d_plots, ax1, ax2, ax3)
            threeplot.plot_main()
            threeplot.set_ylim_by_d(self.d_max_min)
        return fig, axes

    def set_ylim_by_key(self, key, ymin, ymax):
        self.d_max_min[key] = (ymin, ymax)

    def __get_ax123(self, row_id, axes):
        return axes[row_id, 0], axes[row_id, 1], axes[row_id, 2]

    def __read_d_plots_from_pkl(self, host):
        output_folder = path.join(findh_folder, host, 'output')
        fname = path.join(output_folder, 'lp_lncos_deltatheta.pkl')
        with open(fname, 'rb') as handle:
            d_plots = pickle.load(handle)
        return d_plots

    def __initial_d_max_min(self):
        keys = ['<delta theta^2>', 'ln<cos delta theta>', 'lp']
        d = dict()
        for key in keys:
            d[key] = self.__get_max_min_by_key(key)
        return d

    def __get_max_min_by_key(self, key):
        mean_list = list()
        for host in self.hosts:
            d_plots = self.__read_d_plots_from_pkl(host)
            mean_list += d_plots[key]['mean']
        mean_list = np.array(mean_list)
        interval = np.abs(mean_list.max() - mean_list.min()) * 0.05
        minimum = mean_list.min() - interval
        maximum = mean_list.max() + interval
        return (minimum, maximum)


class VarThetaThreePlots:
    lbfz = 14
    lgfz = 12
    ylabel = r'Var$(\theta)$'
    colors = ['r', 'g', 'b', 'cyan', 'magenta', 'orange']
    l_bar = 0.34 # unit: nm, mean of helical rise

    def __init__(self, group1, group2, group3, figsize):
        self.group1 = group1
        self.group2 = group2
        self.group3 = group3
        self.figsize = figsize
    
    def plot_main(self):
        fig, axes = plt.subplots(ncols=3, nrows=1, figsize=self.figsize)
        self.__plot_var_theta(axes[0], self.group1)
        self.__plot_var_theta(axes[1], self.group2)
        self.__plot_var_theta(axes[2], self.group3)
        return fig, axes

    def __plot_var_theta(self, ax, group):
        for idx, host in enumerate(group):
            d_plots = self.__read_d_plots_from_pkl(host)
            xlist, y_mean, y_std = self.__get_xlist_ymean_ystd(d_plots)
            ax.errorbar(xlist, y_mean, yerr=y_std, label=host, linestyle='-', marker='.', color=self.colors[idx], ecolor='grey', capsize=6)
            self.__set_xy_labels(ax)
        #ax.set_xticks(xlist)
        ax.legend(fontsize=self.lgfz, frameon=False)

    def __read_d_plots_from_pkl(self, host):
        output_folder = path.join(findh_folder, host, 'output')
        fname = path.join(output_folder, 'lp_lncos_deltatheta.pkl')
        with open(fname, 'rb') as handle:
            d_plots = pickle.load(handle)
        return d_plots

    def __get_xlist_ymean_ystd(self, d_plots):
        key = '<delta theta^2>'
        xlist = np.array(d_plots['j-i']) * self.l_bar
        y_mean = d_plots[key]['mean']
        y_std = d_plots[key]['std']
        return xlist, y_mean, y_std

    def __set_xy_labels(self, ax):
        ax.set_ylabel(self.ylabel, fontsize=self.lbfz)
        ax.set_xlabel(r'$(j-i)\overline{l}$ (nm)', fontsize=self.lbfz)


class LpSlopesPlots:
    lbfz = 14
    lgfz = 12
    ylabel = r'$< (\theta/2)^2 >$'
    colors = ['r', 'g', 'b', 'cyan', 'magenta', 'orange']
    l_bar = 0.34 # unit: nm, mean of helical rise
    

    def __init__(self, hosts, figsize):
        self.hosts = hosts
        self.n_host = len(hosts)
        self.figsize = figsize
    
    def plot_main(self):
        ncols, nrows = self.__get_ncols_nrows()
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=self.figsize)
        d_axes = self.__get_d_axes(axes)
        for idx, host in enumerate(self.hosts):
            ax = d_axes[idx]
            d_plots = self.__read_d_plots_from_pkl(host)
            self.__plot_ax(ax, d_plots, host, idx)
        self.__set_ylim_for_all_ax(d_axes)
        return fig, axes

    def __get_ncols_nrows(self):
        d_ncols = {2: 2, 6: 3, 4: 2}
        d_nrows = {2: 1, 6: 2, 4: 2}
        return d_ncols[self.n_host], d_nrows[self.n_host]

    def __get_d_axes(self, axes):
        d_axes = dict()
        if self.n_host == 2:
            for idx, ax in enumerate(axes):
                d_axes[idx] = ax
        else:
            idx = 0
            for row_axes in axes:
                for ax in row_axes:
                    d_axes[idx] = ax
                    idx += 1
        return d_axes
        
    def __plot_ax(self, ax, d_plots, host, idx):
        xlist, y_mean, y_std = self.__get_xlist_ymean_ystd(d_plots)
        ax.errorbar(xlist, y_mean, yerr=y_std, label=host, linestyle='-', marker='.', color=self.colors[idx], ecolor='grey', capsize=6)
        self.__set_xy_labels(ax)
        ax.legend(fontsize=self.lgfz, frameon=False)
        lp, rvalue = self.__get_lp_m_rsquare(xlist, y_mean)
        txt = r'$L_p={' + f'{lp:.2f}' + r'}$ nm  ${r=' + f'{rvalue:.4f}' + r'}$'
        xpos, ypos = self.__get_txt_xy(ax, 60, 50)
        ax.text(xpos, ypos, txt, fontsize=self.lgfz)

    def __read_d_plots_from_pkl(self, host):
        output_folder = path.join(findh_folder, host, 'output')
        fname = path.join(output_folder, 'lp_lncos_deltatheta.pkl')
        with open(fname, 'rb') as handle:
            d_plots = pickle.load(handle)
        return d_plots

    def __get_xlist_ymean_ystd(self, d_plots):
        key = '<(theta/2)^2>'
        xlist = np.array(d_plots['j-i']) * self.l_bar
        y_mean = d_plots[key]['mean']
        y_std = d_plots[key]['std']
        return xlist, y_mean, y_std

    def __set_xy_labels(self, ax):
        ax.set_ylabel(self.ylabel, fontsize=self.lbfz)
        ax.set_xlabel(r'$(j-i)\overline{l}$ (nm)', fontsize=self.lbfz)

    def __get_lp_m_rsquare(self, xlist, ylist):
        regre = stats.linregress(xlist, ylist)
        lp = 1 / ( 2 * regre.slope)
        return lp, regre.rvalue

    def __get_txt_xy(self, ax, x_percent, y_percent):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_interval = np.abs(xlim[1] - xlim[0]) / 100
        y_interval = np.abs(ylim[1] - ylim[0]) / 100
        x = xlim[0] + x_percent * x_interval
        y = ylim[0] + y_percent * y_interval
        return x, y

    def __set_ylim_for_all_ax(self, d_axes):
        ymin, ymax = self.__get_ylim(d_axes)
        for idx in range(self.n_host):
            ax = d_axes[idx]
            ax.set_ylim(ymin, ymax)

    def __get_ylim(self, d_axes):
        ymin_list = list()
        ymax_list = list()
        for key in d_axes:
            ax = d_axes[key]
            ylim = ax.get_ylim()
            ymin_list.append(ylim[0])
            ymax_list.append(ylim[1])
        return np.min(ymin_list), np.max(ymax_list)

class LnCosPlots:
    lbfz = 14
    lgfz = 12
    ylabel = r'$\ln{(\left< \cos{\Delta \theta_{ij}}\right>)}$'
    colors = ['r', 'g', 'b', 'cyan', 'magenta', 'orange']
    l_bar = 0.34 # unit: nm, mean of helical rise
    

    def __init__(self, hosts, figsize):
        self.hosts = hosts
        self.n_host = len(hosts)
        self.figsize = figsize
    
    def plot_main(self):
        ncols, nrows = self.__get_ncols_nrows()
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=self.figsize)
        d_axes = self.__get_d_axes(axes)
        for idx, host in enumerate(self.hosts):
            ax = d_axes[idx]
            d_plots = self.__read_d_plots_from_pkl(host)
            self.__plot_ax(ax, d_plots, host, idx)
        self.__set_ylim_for_all_ax(d_axes)
        return fig, axes

    def __get_ncols_nrows(self):
        d_ncols = {2: 2, 6: 3, 4: 2}
        d_nrows = {2: 1, 6: 2, 4: 2}
        return d_ncols[self.n_host], d_nrows[self.n_host]

    def __get_d_axes(self, axes):
        d_axes = dict()
        if self.n_host == 2:
            for idx, ax in enumerate(axes):
                d_axes[idx] = ax
        else:
            idx = 0
            for row_axes in axes:
                for ax in row_axes:
                    d_axes[idx] = ax
                    idx += 1
        return d_axes
        
    def __plot_ax(self, ax, d_plots, host, idx):
        xlist, y_mean, y_std = self.__get_xlist_ymean_ystd(d_plots)
        ax.errorbar(xlist, y_mean, yerr=y_std, label=host, linestyle='-', marker='.', color=self.colors[idx], ecolor='grey', capsize=6)
        self.__set_xy_labels(ax)
        ax.legend(fontsize=self.lgfz, frameon=False)
        lp, rvalue = self.__get_lp_m_rsquare(xlist, y_mean)
        txt = r'$L_p={' + f'{lp:.2f}' + r'}$ nm  ${r=' + f'{rvalue:.4f}' + r'}$'
        xpos, ypos = self.__get_txt_xy(ax, 60, 50)
        ax.text(xpos, ypos, txt, fontsize=self.lgfz)

    def __read_d_plots_from_pkl(self, host):
        output_folder = path.join(findh_folder, host, 'output')
        fname = path.join(output_folder, 'lp_lncos_deltatheta.pkl')
        with open(fname, 'rb') as handle:
            d_plots = pickle.load(handle)
        return d_plots

    def __get_xlist_ymean_ystd(self, d_plots):
        key = 'ln<cos delta theta>'
        xlist = np.array(d_plots['j-i']) * self.l_bar
        y_mean = d_plots[key]['mean']
        y_std = d_plots[key]['std']
        return xlist, y_mean, y_std

    def __set_xy_labels(self, ax):
        ax.set_ylabel(self.ylabel, fontsize=self.lbfz)
        ax.set_xlabel(r'$(j-i)\overline{l}$ (nm)', fontsize=self.lbfz)

    def __get_lp_m_rsquare(self, xlist, ylist):
        regre = stats.linregress(xlist, ylist)
        lp = -1 / regre.slope
        return lp, regre.rvalue

    def __get_txt_xy(self, ax, x_percent, y_percent):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_interval = np.abs(xlim[1] - xlim[0]) / 100
        y_interval = np.abs(ylim[1] - ylim[0]) / 100
        x = xlim[0] + x_percent * x_interval
        y = ylim[0] + y_percent * y_interval
        return x, y

    def __set_ylim_for_all_ax(self, d_axes):
        ymin, ymax = self.__get_ylim(d_axes)
        for idx in range(self.n_host):
            ax = d_axes[idx]
            ax.set_ylim(ymin, ymax)

    def __get_ylim(self, d_axes):
        ymin_list = list()
        ymax_list = list()
        for key in d_axes:
            ax = d_axes[key]
            ylim = ax.get_ylim()
            ymin_list.append(ylim[0])
            ymax_list.append(ylim[1])
        return np.min(ymin_list), np.max(ymax_list)

def get_pl_stat(df, i, j):
    j_minus_i = j - i
    mask = (df['i'] == i) & (df['j'] == j)
    df_temp = df[mask]
    thetas = df_temp['theta'].values
    square_thetas_div_2_mean = np.mean(np.square(thetas / 2))
    theta_var = thetas.var()
    theta_mean = thetas.mean()
    #delta_theta_array = thetas - theta_mean
    delta_theta_array = thetas
    cos_delta_theta = np.cos(delta_theta_array)
    mean_cos_delta_theta = cos_delta_theta.mean()
    log_mean_cos_delta_theta = np.log(mean_cos_delta_theta)
    Lp = (-j_minus_i * 3.4) / (2 * log_mean_cos_delta_theta)
    lp = Lp / 10
    return theta_var, mean_cos_delta_theta, log_mean_cos_delta_theta, lp, square_thetas_div_2_mean