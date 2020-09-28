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

    def draw_regr_var_an_n(self, ax, select_nlist):
        n_continuous, y_continuous = self.get_continuous_line()
        y = [self.get_var_an_by_n(n) for n in select_nlist]

        ax.scatter(self.nlist, self.var_list)
        ax.plot(n_continuous, y_continuous, '--', color='blue', alpha=0.6)
        ax.plot(select_nlist, y, 'x', color='red')
        ax.set_xlabel('n', fontsize=12)
        ax.set_ylabel('Var($a_n$)', fontsize=12)
        ax.set_title(self.host, fontsize=14)
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