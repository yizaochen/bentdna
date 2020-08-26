from os import path
import pandas as pd
import numpy as np

class DataFrameGenerator:

    columns = ['frame_id', 's', 'theta(s)', 'label', 'a_n']
    fname_suffix = {'MD': 'MD', 'n=0to9': 'n0to9', 'n=1': 'n1',
                    'n=2': 'n2', 'n=3': 'n3', 'n=4': 'n4', 'n=5': 'n5',
                    'n=6': 'n6', 'n=7': 'n7', 'n=8': 'n8', 'n=9': 'n9'}

    def __init__(self, workfolder, host):
        self.host = host
        self.rootfolder = path.join(workfolder, host)
        self.an_folder = path.join(self.rootfolder, 'an_folder')

    def make_df(self, label, s_agent, last_frame):
        df_filename = self.__get_df_name(label)
        d_result = self.__initialize_d_result()

        # Process Data
        if label == 'MD':
            d_result = self.__process_md_data(label, d_result, last_frame, s_agent)
        elif label == 'n=0to9':
            d_result = self.__process_fourier_approximate(label, d_result, last_frame, s_agent)
        else:
            n = int(label[-1])
            d_result = self.__process_fourier_singlemode(label, d_result, last_frame, s_agent, n)

        df = pd.DataFrame(d_result)
        df = df[self.columns]
        df.to_csv(df_filename)
        print(f'Write DataFrame to {df_filename}')
        return df

    def read_df(self, label):
        df_filename = self.__get_df_name(label)
        df = pd.read_csv(df_filename)
        print(f'Read DataFrame from {df_filename}')
        return df

    def __process_md_data(self, label, d_result, last_frame, s_agent):
        s_list, theta_list = s_agent.get_slist_thetalist(0)
        n_nodes = len(s_list)
        d_result = self.__initialize_dimension(n_nodes, last_frame, d_result, label)
        i = 0
        for frame_id in range(1, last_frame+1):
            j = i + n_nodes
            s_list, theta_list = s_agent.get_slist_thetalist(frame_id)
            d_result['s'][i:j] = s_list
            d_result['theta(s)'][i:j] = theta_list
            d_result['frame_id'][i:j] = frame_id
            i += n_nodes
        return d_result

    def __process_fourier_approximate(self, label, d_result, last_frame, s_agent):
        n_begin = 0
        n_end = 9
        s_list, theta_list = s_agent.get_slist_thetalist(0)
        n_nodes = len(s_list)
        d_result = self.__initialize_dimension(n_nodes, last_frame, d_result, label)
        i = 0
        for frame_id in range(1, last_frame+1):
            j = i + n_nodes
            s_list, theta_list = s_agent.get_approximate_theta(frame_id, n_begin, n_end)
            d_result['s'][i:j] = s_list
            d_result['theta(s)'][i:j] = theta_list
            d_result['frame_id'][i:j] = frame_id
            i += n_nodes
        return d_result

    def __process_fourier_singlemode(self, label, d_result, last_frame, s_agent, n):
        s_list, theta_list = s_agent.get_slist_thetalist(0)
        n_nodes = len(s_list)
        d_result = self.__initialize_dimension(n_nodes, last_frame, d_result, label)
        i = 0
        for frame_id in range(1, last_frame+1):
            j = i + n_nodes
            s_list, theta_list, an = s_agent.get_approximate_theta_singlemode(frame_id, n)
            d_result['s'][i:j] = s_list
            d_result['theta(s)'][i:j] = theta_list
            d_result['frame_id'][i:j] = frame_id
            d_result['a_n'][i:j] = an
            i += n_nodes
        return d_result

    def __get_df_name(self, label):
        suffix = self.fname_suffix[label]
        return path.join(self.an_folder, f'animation_{suffix}.csv')

    def __initialize_d_result(self):
        d_result = dict()
        for column in self.columns:
            d_result[column] = list()
        return d_result

    def __initialize_dimension(self, n_nodes, last_frame, d_result, label):
        n_dim = n_nodes * last_frame
        d_result['frame_id'] = np.zeros(n_dim, dtype=int)
        d_result['s'] = np.zeros(n_dim)
        d_result['theta(s)'] = np.zeros(n_dim)
        d_result['label'] = np.zeros(n_dim, dtype=object)
        d_result['label'][:] = label
        d_result['a_n'] = np.zeros(n_dim)
        return d_result
