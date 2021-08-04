from os import path, system

hosts = [ 'atat_21mer', 'g_tract_21mer', 'a_tract_21mer', 'gcgc_21mer',
          'ctct_21mer', 'tgtg_21mer', 'tat_21mer', 'tat_1_21mer', 'tat_2_21mer', 'tat_3_21mer']

rootfolder = '/home/yizaochen/codes/dna_rna/all_systems'
fhelix_folder = '/home/yizaochen/codes/dna_rna/length_effect/find_helical_axis'

for host in hosts:
    allatom_folder = path.join(rootfolder, host, 'bdna+bdna', 'input', 'allatoms')
    f_input_folder = path.join(fhelix_folder, host, 'input')
    old_dcd = path.join(allatom_folder, 'bdna+bdna.central.dcd')
    new_dcd = path.join(f_input_folder, 'bdna+bdna.0_5000ns.50000frames.dcd')
    cmd = f'cp {old_dcd} {new_dcd}'
    system(cmd)
    print(cmd)


