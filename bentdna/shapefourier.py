from os import path
import pandas as pd


class ShapeAgent:
    d_n_bp = {
        'atat_21mer': 21, 'g_tract_21mer': 21, 'a_tract_21mer': 21,
        'yizao_model': 24, 'pnas_16mer': 16, 'gcgc_21mer': 21,
        'ctct_21mer': 21, 'tgtg_21mer': 21, '500mm': 16,
        'only_cation': 16, 'mgcl2_150mm': 16 }

    def __init__(self, workfolder, host):
        self.host = host
        self.rootfolder = path.join(workfolder, host)
        self.df_folder = path.join(self.rootfolder, 'l_theta')
        self.n_bead = self.d_n_bp[host]
        self.df_name = path.join(self.df_folder, f'l_modulus_theta_{self.n_bead}_beads.csv')
        self.df = None
        
    def read_l_modulus_theta(self):
        self.df =  pd.read_csv(self.df_name)