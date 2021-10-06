import matplotlib.pyplot as plt
from bentdna.persistence_length import LpWindows

class BarPlot:
    hosts = ['a_tract_21mer', 'g_tract_21mer', 'atat_21mer', 'gcgc_21mer']
    d_color = {'a_tract_21mer': '#5C8ECB', 'g_tract_21mer': '#EA6051', 'atat_21mer': '#8CF8D5', 'gcgc_21mer': '#E75F93'}
    d_abbr = {'a_tract_21mer': 'A-tract', 'atat_21mer': 'TATA', 
              'g_tract_21mer': 'G-tract', 'gcgc_21mer': 'CpG'}
    n_frames_per_window = 5000
    tickfz_y = 4
    tickfz_x = 4
    lbfz = 5
    lgfz = 5

    def __init__(self, bentna_folder):
        self.bentna_folder = bentna_folder
        self.d_l_agent = self.get_d_l_agent()
        self.d_x = self.get_d_x()
        self.d_mean, self.d_std = self.get_d_mean_std()

        self.n_hosts = len(self.hosts)
        self.xticklabels = [self.d_abbr[host] for host in self.hosts]

    def bar_plot(self, figsize, width=0.4):
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        for host in self.hosts:
            self.bar_plot_host(ax, host, width)
        self.set_xtick_xlabel(ax)
        ax.set_ylabel(r'$\mathrm{L}_\mathrm{p}$ (nm)', fontsize=self.lbfz, labelpad=1.0)
        self.plot_experimental_value(ax)
        return fig, ax

    def plot_experimental_value(self, ax):
        exp_value = 50
        ax.axhline(exp_value, linestyle='--', linewidth=0.5, color='magenta', alpha=0.5, label=r'Experimental $\mathrm{L}_\mathrm{p}$')
        ax.legend(fontsize=self.lgfz, frameon=False)

    def set_xtick_xlabel(self, ax):
        ax.set_xticks([1,2,4,5])
        ax.set_xticklabels(self.xticklabels)
        ax.set_yticks(range(0,71,10))
        ax.tick_params(axis='y', labelsize=self.tickfz_y, length=1.5, pad=1)
        ax.tick_params(axis='x', labelsize=self.tickfz_x, length=1.5, pad=1)

    def bar_plot_host(self, ax, host, width):
        color = self.d_color[host]
        ax.bar(self.d_x[host], self.d_mean[host], yerr=self.d_std[host], color=color, ecolor='black')

    def get_d_l_agent(self):
        d_l_agent = dict()
        for host in self.hosts:
            d_l_agent[host] = LpWindows(self.bentna_folder, host, self.n_frames_per_window)
            d_l_agent[host].read_lp_store_array()
        return d_l_agent

    def get_d_mean_std(self):
        n_begin = 2
        n_end = 2
        d_mean = dict()
        d_std = dict()
        for host in self.hosts:
            pack_data = self.d_l_agent[host].get_nlist_mean_std_array_for_windows(n_begin, n_end)
            d_mean[host] = pack_data[1][0]
            d_std[host] = pack_data[2][0]
        return d_mean, d_std

    def get_d_x(self):
        return {'a_tract_21mer': 1, 'g_tract_21mer': 2, 'atat_21mer': 4, 'gcgc_21mer': 5}


    