from os import path
from bentdna.miscell import check_dir_exist_and_make

rootfolder = '/home/yizaochen/codes/dna_rna/length_effect/find_helical_axis'
allsystems = ['atat_21mer', 'a_tract_21mer', 'gcgc_21mer', 'g_tract_21mer', 'ctct_21mer', 'tgtg_21mer']

for host in allsystems:
    folder = path.join(rootfolder, host)
    check_dir_exist_and_make(folder)
