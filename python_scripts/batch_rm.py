from os import path, system

hosts = ['atat_21mer', 'g_tract_21mer', 'a_tract_21mer', 'gcgc_21mer',
         'ctct_21mer', 'tgtg_21mer', 'tat_21mer', 'tat_1_21mer', 'tat_2_21mer', 'tat_3_21mer']

fhelix_folder = '/home/yizaochen/codes/dna_rna/length_effect/find_helical_axis'

for host in hosts:
    f_input_folder = path.join(fhelix_folder, host, 'input')
    f_wait_rm = path.join(f_input_folder, 'bdna+bdna.all.xtc')
    if path.exists(f_wait_rm):
        print(f'{f_wait_rm} exists')
        cmd = f'rm {f_wait_rm}'
        #system(cmd)
        print(cmd)
    else:
        print(f'{f_wait_rm} not exist')
