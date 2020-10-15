from os import path
from bentdna.find_haxis_curve import FindHelixAgent, PrepareHelix

allsystems = ['only_cation', 'mgcl2_150mm']
findhelix_folder = '/home/vortex_yizaochen/yizaochen/codes/dna_rna/length_effect/find_helical_axis'

n_bp = 16
for host in allsystems:
    prep_helix = PrepareHelix(findhelix_folder, host, n_bp)
    f_agent = FindHelixAgent(prep_helix.workfolder, prep_helix.pdb_modi, prep_helix.dcd_out, n_bp)
    f_agent.extract_pdb_allatoms()
    f_agent.curveplus_find_haxis()
