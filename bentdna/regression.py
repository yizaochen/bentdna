import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from bentdna.shapefourier import ShapeAgent

class RegressAgent:

    n_begin = 0
    n_end = 9
    nlist = range(1, 10)

    def __init__(self, host, workfolder):
        self.host = host
        self.workfolder = workfolder

        self.df_an = self.__read_df_an()

        self.x_all = self.nlist_to_x(self.nlist)
        self.var_list = self.get_var_list() # y_all

        self.regr = LinearRegression()
        self.a = None

    def draw_observe_plot(self):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,4))
        ax = axes[0]
        ax.scatter(self.nlist, self.var_list)
        ax.set_xlabel('n', fontsize=12)
        ax.set_ylabel('Var($a_n$)', fontsize=12)
        ax.set_title(self.host, fontsize=14)
        
        ax = axes[1]
        ax.scatter(self.x_all, self.var_list)
        ax.set_xlabel('$(1/n)^2$', fontsize=12)
        ax.set_ylabel('Var($a_n$)', fontsize=12)
        return fig, axes

    def draw_regr_var_an_nsquare_inverse(self, select_nlist):
        x = self.nlist_to_x(select_nlist)
        y = [self.get_var_an_by_n(n) for n in select_nlist]
        X_all = self.reshape_array(self.x_all)
        y_pred = self.regr.predict(X_all)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,4))
        ax.scatter(self.x_all, self.var_list,  color='black')
        ax.plot(self.x_all, y_pred, color='blue', linewidth=3)
        ax.plot(x, y, 'x', color='red')
        ax.set_xlabel('$(1/n)^2$', fontsize=12)
        ax.set_ylabel('Var($a_n$)', fontsize=12)
        return fig, ax

    def draw_regr_var_an_n(self, ax, select_nlist, host_abbr, showxylabel, markersize, ylim, xratio, yratio):
        showxlabel, showylabel = showxylabel
        n_continuous, y_continuous = self.get_continuous_line()
        y = [self.get_var_an_by_n(n) for n in select_nlist]

        ax.scatter(self.nlist, self.var_list, s=markersize)
        ax.plot(n_continuous, y_continuous, '--', linewidth=1, color='blue', alpha=0.6)
        ax.plot(select_nlist, y, 'x', color='red', markersize=markersize)
        if showxlabel:
            ax.set_xlabel('n', fontsize=12)
        if showylabel:
            ax.set_ylabel('Var($a_n$)', fontsize=12)
        ax.set_xticks(self.nlist)
        ax.set_ylim(ylim)
        ax.tick_params(axis='both', labelsize=10)
        self.annotate_host(ax, host_abbr, xratio, yratio)
        return ax

    def get_continuous_line(self):
        n = np.linspace(1, 9, num=50)
        x = self.nlist_to_x(n)
        X = self.reshape_array(x)
        y = self.regr.predict(X)
        return n, y

    def do_regression(self, select_nlist):
        X, Y = self.__get_X_Y(select_nlist)
        self.regr.fit(X, Y)

        X_all = self.reshape_array(self.x_all)
        Y_all = self.reshape_array(self.var_list)
        y_pred = self.regr.predict(X_all)

        self.a = self.regr.coef_[0][0]
        mse = mean_squared_error(Y_all, y_pred)
        r2 = r2_score(Y_all, y_pred)
        print(f'Coefficients: {self.a:.3f}')
        print(f'Mean squared error: {mse:.3f}')
        print(f'Coefficient of determination: {r2:.2f}')

    def get_var_an_by_n(self, n):
        key = str(n)
        return self.df_an[key].var()

    def get_var_list(self):
        return [self.get_var_an_by_n(n) for n in self.nlist]

    def __read_df_an(self):
        s_agent = ShapeAgent(self.workfolder, self.host)
        return s_agent.read_df_an(self.n_begin, self.n_end)

    def __get_X_Y(self, select_nlist):
        x = self.nlist_to_x(select_nlist)
        y = [self.get_var_an_by_n(n) for n in select_nlist]
        X = self.reshape_array(x)
        Y = self.reshape_array(y)
        return X, Y

    def nlist_to_x(self, nlist):
        x = np.array([1/n for n in nlist])
        return np.square(x)

    def reshape_array(self, x):
        return np.matrix(x).reshape(len(x), 1)
        
    def annotate_host(self, ax, host, xratio, yratio):
        x_text, y_text = self.get_xtext_ytext_for_sci_anotation(ax, xratio, yratio)
        ax.text(x_text, y_text, host, fontsize=12)
        
    def get_xtext_ytext_for_sci_anotation(self, ax, xratio, yratio):
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        n_portion = 100
        x_unit = (xlim[1] - xlim[0]) / n_portion
        y_unit = (ylim[1] - ylim[0]) / n_portion
        x_text = xlim[0] + xratio * x_unit
        y_text = ylim[0] + yratio * y_unit
        return x_text, y_text


class SixPlots:
    workfolder = '/home/yizaochen/codes/dna_rna/length_effect/find_helical_axis'
    hosts = ['a_tract_21mer', 'atat_21mer', 'ctct_21mer',
             'g_tract_21mer', 'gcgc_21mer', 'tgtg_21mer']
    abbr_hosts = {'a_tract_21mer': 'A-tract', 'ctct_21mer': 'CTCT', 'gcgc_21mer': 'GCGC',
                  'g_tract_21mer': 'G-tract', 'atat_21mer': 'ATAT', 'tgtg_21mer': 'TGTG'}
    d_xylabel = {'a_tract_21mer': (False, True), 'ctct_21mer': (False, False), 'gcgc_21mer': (True, False),
                 'g_tract_21mer': (True, True), 'atat_21mer': (False, False), 'tgtg_21mer': (True, False)}
    nrows = 2
    ncols = 3
    select_nlist = [2, 3, 4] # Ad hoc

    def __init__(self, figsize, markersize, ylim, xratio, yratio):
        self.figsize = figsize
        self.markersize = markersize
        self.ylim = ylim
        self.xratio = xratio
        self.yratio = yratio

    def plot_main(self):
        fig, axes = plt.subplots(nrows=self.nrows, ncols=self.ncols, figsize=self.figsize)
        d_axes = self.get_d_axes(axes)
        for host in self.hosts:
            ax = d_axes[host]
            temp_r_agent = RegressAgent(host, self.workfolder)
            temp_r_agent.do_regression(self.select_nlist)
            temp_r_agent.draw_regr_var_an_n(ax, self.select_nlist, self.abbr_hosts[host], 
                                            self.d_xylabel[host], self.markersize, self.ylim, 
                                            self.xratio, self.yratio)
        return fig, d_axes

    def get_d_axes(self, axes):
        d_axes = dict()
        host_id = 0
        for row_id in range(self.nrows):
            for col_id in range(self.ncols):
                host = self.hosts[host_id]
                d_axes[host] = axes[row_id, col_id]
                host_id += 1
        return d_axes