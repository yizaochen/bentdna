from os import path
from bentdna.miscell import check_dir_exist_and_make

rootfolder = '/home/yizaochen/Desktop/methyl_dna'
allsystems = ['cg_13_meth1']

for host in allsystems:
    folder = path.join(rootfolder, host)
    check_dir_exist_and_make(folder)
