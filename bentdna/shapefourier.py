from os import path, rename
import pandas as pd
import numpy as np
from bentdna.miscell import check_dir_exist_and_make

class ShapeAgent:
    d_n_bp = {
        'atat_21mer': 21, 'g_tract_21mer': 21, 'a_tract_21mer': 21,
        'yizao_model': 24, 'pnas_16mer': 16, 'gcgc_21mer': 21,
        'ctct_21mer': 21, 'tgtg_21mer': 21, '500mm': 16,
        'only_cation': 16, 'mgcl2_150mm': 16 }
        
    start_end = {
        'atat_21mer': (3, 17), 'g_tract_21mer': (3, 17), 
        'a_tract_21mer': (3, 17), 'yizao_model': (3, 20), 
        'pnas_16mer': (3, 12), 'gcgc_21mer': (3, 17),
        'ctct_21mer': (3, 17), 'tgtg_21mer': (3, 17), '500mm': (3, 12),
        'only_cation': (3, 12), 'mgcl2_150mm': (3, 12)}

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