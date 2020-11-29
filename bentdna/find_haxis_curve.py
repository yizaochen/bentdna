from os import path, remove, rename
from shutil import copyfile
from glob import glob
import subprocess
import MDAnalysis as mda
from bentdna.miscell import check_dir_exist_and_make
from bentdna.atom import Atom
from bentdna.PDB import PDBReader, PDBWriter

def copy_file(f1, f2):
    copyfile(f1, f2)
    print(f'cp {f1} {f2}')


class AvgAgent:
    all_folder = '/home/yizaochen/codes/dna_rna/all_systems'
    type_na = 'bdna+bdna'
    n_bp = 21

    def __init__(self, findhelix_folder, host):
        self.host = host
        self.host_folder = path.join(self.all_folder, host)
        self.na_folder = path.join(self.host_folder, 'bdna+bdna')
        self.all_input_folder = path.join(self.na_folder, 'input')
        self.heavy_folder = path.join(self.all_input_folder, 'heavyatoms')

        self.avg_crd = path.join(self.heavy_folder, f'{self.type_na}.nohydrogen.avg.crd')
        self.avg_pdb = path.join(self.heavy_folder, f'{self.type_na}.nohydrogen.avg.pdb')
        self.avg_pdb_backup = path.join(self.heavy_folder, f'{self.type_na}.nohydrogen.avg.backup.pdb')

        self.findhelix_folder = findhelix_folder
        self.workfolder = path.join(self.findhelix_folder, host)
        self.avg_folder = path.join(self.workfolder, 'avg_structure')
        self.input_folder = path.join(self.workfolder, 'input')
        self.pdb_modi = path.join(self.input_folder, 'bdna_modi_avg.pdb')

        self.haxis_pdb = path.join(self.avg_folder, 'haxis.avg.pdb')
        self.smooth_pdb = path.join(self.avg_folder, 'haxis.smooth.avg.pdb')

        self.__check_and_make_folders()

    def convert_crd_to_pdb(self):
        u = mda.Universe(self.avg_crd, self.avg_crd)
        with mda.Writer(self.avg_pdb, bonds=None, n_atoms=u.atoms.n_atoms) as PDBOUT:
            PDBOUT.write(u.atoms)
        print(f'Convert {self.avg_crd} to {self.avg_pdb}')
        print('Check by:')
        print(f'vmd -pdb {self.avg_pdb}')

    def backup_avg_pdb(self):
        copy_file(self.avg_pdb, self.avg_pdb_backup)

    def cpback_avg_pdb(self):
        # This is used in rescue!!!
        print('When rescuing, the command is:')
        print(f'cp {self.avg_pdb_backup} {self.avg_pdb}')

    def change_resid_to_modi(self):
        reader = PDBReader(self.avg_pdb, segid_exist=True)
        atgs = reader.get_atomgroup()

        for atom in atgs:
            if atom.segid == 'B':
                atom.resid += self.n_bp
                
        writer = PDBWriter(self.pdb_modi, atgs)
        writer.write_pdb()

    def curveplus_find_haxis(self):
        lis_name = 'r+bdna'
        frame_id = 'avg'
        agent = CurvePlusAgent(self.pdb_modi, self.workfolder, self.n_bp, lis_name, frame_id, self.avg_folder, self.avg_folder)
        agent.clean_files()
        agent.execute_curve_plus()
        agent.extract_haxis_to_pdb()
        agent.get_smooth_haxis()

    def vmd_check(self):
        cmd = 'cd /home/yizaochen/codes/bentdna'
        print(cmd)
        cmd = f'vmd -pdb {self.avg_pdb}'
        print(cmd)
        cmd = f'mol new {self.haxis_pdb} type pdb'
        print(cmd)
        cmd = f'mol new {self.smooth_pdb} type pdb'
        print(cmd)
        cmd = f'source ./tcl/draw_aa_haxis.tcl'
        print(cmd)

    def __check_and_make_folders(self):
        for folder in [self.input_folder, self.avg_folder]:
            check_dir_exist_and_make(folder)


class PrepareHelix:
    all_folder = '/home/yizaochen/codes/dna_rna/all_systems'
    type_na = 'bdna+bdna'
    d_lastframe = {
        'atat_21mer': 10000, 'g_tract_21mer': 10000, 'a_tract_21mer': 10000,
        'yizao_model': 20000, 'pnas_16mer': 10000, 'gcgc_21mer': 10000,
        'ctct_21mer': 10000, 'tgtg_21mer': 10000, '500mm': 10000,
        'only_cation': 10000, 'mgcl2_150mm': 10000, 
        'cg_13_meth1': 10000 }

    def __init__(self, findhelix_folder, host, n_bp):
        self.findhelix_folder = findhelix_folder
        self.host = host
        self.n_bp = n_bp
        self.workfolder = path.join(self.findhelix_folder, host)
        self.input_folder = path.join(self.workfolder, 'input')
        self.output_folder = path.join(self.workfolder, 'output')
        self.pdb_in = path.join(self.input_folder, f'{self.type_na}.npt4.all.pdb')
        self.xtc_in = path.join(self.input_folder, f'{self.type_na}.all.xtc')
        self.pdb_modi = path.join(self.input_folder, 'bdna_modi.pdb')

        self.lastframe = self.d_lastframe[host]
        self.dcd_out = self.__get_dcd_out()
        self.dcd_out_test = self.__get_dcd_out_test()
        self.all_na_folder = path.join(self.all_folder, host, self.type_na)
        self.__check_and_make_folders()

    def copy_input_xtc(self):
        xtcname = f'{self.type_na}.all.xtc'
        target_f = path.join(self.all_na_folder, 'input/allatoms', xtcname)
        copy_file(target_f, self.xtc_in)

    def copy_input_pdb(self):
        pdbname = f'{self.type_na}.npt4.all.pdb'
        target_f = path.join(self.all_na_folder, 'input/allatoms', pdbname)
        copy_file(target_f, self.pdb_in)

    def copy_pdb_to_pdbmodi(self):
        if path.exists(self.pdb_modi):
            print(f'{self.pdb_modi} already exists!')
        else:
            copy_file(self.pdb_in, self.pdb_modi)

    def move_h0pdb_to_outfolder(self):
        h0_pdb = path.join(self.workfolder, 'pdbs_haxis', 'haxis.0.pdb')
        h0_pdb_out = path.join(self.output_folder, 'haxis.0.pdb')
        if path.exists(h0_pdb):
            rename(h0_pdb, h0_pdb_out)
            print(f'mv {h0_pdb} {h0_pdb_out}')
        else:
            print(f'{h0_pdb} not exist !!!')

    def change_resid_to_modi(self):
        reader = PDBReader(self.pdb_in, segid_exist=True)
        atgs = reader.get_atomgroup()

        for atom in atgs:
            if atom.segid == 'B':
                atom.resid += self.n_bp
                
        writer = PDBWriter(self.pdb_modi, atgs)
        writer.write_pdb()

    def __get_dcd_out(self):
        last_time = f'{self.lastframe/10:.0f}'
        fname = f'{self.type_na}.0_{last_time}ns.{self.lastframe}frames.dcd'
        return path.join(self.input_folder, fname)

    def __get_dcd_out_test(self):
        fname = f'{self.type_na}.0_1ns.10frames.dcd'
        return path.join(self.input_folder, fname)

    def __check_and_make_folders(self):
        for folder in [self.input_folder, self.output_folder]:
            check_dir_exist_and_make(folder)



class FindHelixAgent:
    """
    start_frame: default=0
    stop_frame: default=n_frame
    """
    def __init__(self, rootfolder, pdb_in, dcd_in, n_bp, start_frame=0, stop_frame=None):
        self.n_bp = n_bp
        self.pdb_in = pdb_in
        self.dcd_in = dcd_in
        self.u = mda.Universe(pdb_in, dcd_in)

        self.n_frames = len(self.u.trajectory)     
        self.start_frame = start_frame
        self.stop_frame = self.__get_stop_frame(stop_frame)

        self.pdb_all_folder = path.join(rootfolder, 'pdbs_allatoms')
        self.workfolder = path.join(rootfolder, f'curve_workdir_{self.start_frame}_{self.stop_frame}')
        self.pdb_haxis_folder = path.join(rootfolder, 'pdbs_haxis')
        self.pdb_h_smooth_folder = path.join(rootfolder, 'haxis_smooth')

        self.__check_and_make_folders()
        self.__print_frame()
        
    def extract_pdb_allatoms(self):
        agent = ExtractPDBAgent(self.pdb_in, self.dcd_in, self.pdb_all_folder, self.start_frame, self.stop_frame)
        agent.extract_pdb_from_dcd()

    def curveplus_find_haxis(self):
        lis_name = 'r+bdna'
        for frame_id in range(self.start_frame, self.stop_frame):
            single_pdb = path.join(self.pdb_all_folder, f'{frame_id}.pdb')
            agent = CurvePlusAgent(single_pdb, self.workfolder, self.n_bp, lis_name, frame_id, self.pdb_haxis_folder, self.pdb_h_smooth_folder)
            agent.clean_files()
            agent.execute_curve_plus()
            agent.extract_haxis_to_pdb()

    def curveplus_find_smooth_haxis(self):
        lis_name = 'r+bdna'
        for frame_id in range(self.start_frame, self.stop_frame):
            single_pdb = path.join(self.pdb_all_folder, f'{frame_id}.pdb')
            agent = CurvePlusAgent(single_pdb, self.workfolder, self.n_bp, lis_name, frame_id, self.pdb_haxis_folder, self.pdb_h_smooth_folder)
            agent.clean_files()
            agent.execute_curve_plus()
            agent.get_smooth_haxis()

    def __get_stop_frame(self, stop_frame):
        if stop_frame is None:
            return self.n_frames
        else:
            return stop_frame

    def __print_frame(self):
        print(f'There are {self.n_frames} frames.')

    def __check_and_make_folders(self):
        for folder in [self.pdb_all_folder, self.workfolder, self.pdb_haxis_folder, self.pdb_h_smooth_folder]:
            check_dir_exist_and_make(folder)


class CurvePlusAgent:
    def __init__(self, pdb_in, workfolder, n_bp, lis_name, frame_id, pdb_haxis_folder, pdb_h_smooth_folder):
        self.pdb_in = pdb_in
        self.n_bp = n_bp
        self.lis_name = lis_name
        self.workfolder = workfolder
        self.frame_id = frame_id
        self.pdb_haxis_folder = pdb_haxis_folder
        self.pdb_h_smooth_folder = pdb_h_smooth_folder

    def clean_files(self):
        pathname = path.join(self.workfolder, f'{self.lis_name}*')
        filelist = glob(pathname)
        if len(filelist) != 0:
            for fname in filelist:
                remove(fname)

    def execute_curve_plus(self):
        cmd = self.__get_exectue_curve_plus_cmd()
        errlog = open(path.join(self.workfolder, 'err.log'), 'w')
        outlog = open(path.join(self.workfolder, 'out.log'), 'w')
        subprocess.run(cmd, shell=True, stdout=outlog, stderr=errlog,check=False)
        errlog.close()
        outlog.close()

    def extract_haxis_to_pdb(self):
        axis_pdb = path.join(self.workfolder, f'{self.lis_name}_X.pdb')
        pdb_out = path.join(self.pdb_haxis_folder, f'haxis.{self.frame_id}.pdb')
        agent = ExtractHaxisAgent(self.n_bp, axis_pdb, pdb_out)
        agent.write_pdb()

    def get_smooth_haxis(self):
        axis_pdb = path.join(self.workfolder, f'{self.lis_name}_X.pdb')
        pdb_out = path.join(self.pdb_h_smooth_folder, f'haxis.smooth.{self.frame_id}.pdb')
        copyfile(axis_pdb, pdb_out)
        print(f'cp {axis_pdb} {pdb_out}')

    def __get_exectue_curve_plus_cmd(self):
        curve = '/home/yizaochen/opt/curves+/Cur+'
        inp_end_txt = self.__get_inp_end()
        n1, n2, n3, n4 = self.__get_four_numbers()
        cmd1 = f'{curve} <<!\n'
        cmd2 = f'  &inp {inp_end_txt}&end\n'
        cmd3 = '2 1 -1 0 0\n'
        cmd4 = f'{n1}:{n2}\n'
        cmd5 = f'{n3}:{n4}\n'
        cmd6 = '!'
        return cmd1 + cmd2 + cmd3 + cmd4 + cmd5 + cmd6

    def __get_inp_end(self):
        curve_folder = '/home/yizaochen/opt/curves+/standard'
        lis = path.join(self.workfolder, self.lis_name)
        return f'file={self.pdb_in},\n  lis={lis},\n  lib={curve_folder},'
    
    def __get_four_numbers(self):
        return 1, self.n_bp, 2*self.n_bp, self.n_bp+1
        

class ExtractPDBAgent:
    def __init__(self, pdb_in, dcd_in, out_folder, start_frame, stop_frame):
        self.u = mda.Universe(pdb_in, dcd_in)
        self.out_folder = out_folder

        self.n_frames = len(self.u.trajectory)     
        self.start_frame = start_frame
        self.stop_frame = stop_frame
        
    def extract_pdb_from_dcd(self):
        for ts in self.u.trajectory[self.start_frame:self.stop_frame]:
            pdb_out = path.join(self.out_folder, f'{ts.frame}.pdb')
            self.__process_single_frame(pdb_out)

    def __process_single_frame(self, pdb_out):
        with mda.Writer(pdb_out, bonds=None, n_atoms=self.u.atoms.n_atoms) as PDBOUT:
            PDBOUT.write(self.u.atoms)
        

class ExtractHaxisAgent:
    def __init__(self, n_bp, pdb_in, pdb_out):
        self.u = mda.Universe(pdb_in)
        self.n_bp = n_bp
        self.pdb_out = pdb_out
        self.atg = self.__get_atomgroups()

    def write_pdb(self):
        writer = PDBWriter(self.pdb_out, self.atg)
        writer.write_pdb()

    def __get_atomgroups(self):
        atomgroups = list()
        for resid in range(1, self.n_bp):
            atomgroups.append(self.__get_Atom(resid, False))
        atomgroups.append(self.__get_Atom(resid, True))
        return atomgroups

    def __get_Atom(self, resid, tail):
        if tail:
            atom_data = ['ATOM', resid+1, 'S', 'HAI', resid+1]
        else:
            atom_data = ['ATOM', resid, 'S', 'HAI', resid]
        xyz_list = self.__get_xyz_list(resid, tail)
        atom_data += xyz_list
        atom_data += [1.00, 1.00]
        return Atom(atom_data, False)

    def __get_xyz_list(self, resid, tail):
        atg_sele = self.u.select_atoms(f'resid {resid}')
        if not tail:
            return list(atg_sele.positions[0])
        else:
            return list(atg_sele.positions[-1])
            