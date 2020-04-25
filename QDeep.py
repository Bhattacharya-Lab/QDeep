#!/usr/bin/python
#######################################################################################################
# Name: QDeep.py
# Purpose: Protein single-model quality assessment using Residual Neural Network
# Developed by: Md Hossain Shuvo
# Developed on: 1/17/2020
# Modified by: 
# Change log:
#######################################################################################################
import os, sys, math, time
import argparse, subprocess
import numpy as np
from tensorflow.keras.models import model_from_json
from datetime import timedelta
start_time = time.monotonic()
#----------------------global variables-------------------------#
#                                                               #
#---------------------------------------------------------------#
features = []
finalFeatures = []
window_size = 0
seq_length = 0

print('\n***************************************************************************')
print('*                               QDeep                                     *')
print('*   Distance-based protein model quality estimation using deep ResNets    *')
print('*          For comments, please email to bhattacharyad@auburn.edu         *')
print('***************************************************************************')

#------------------------configure------------------------------#
#Configures before the first use                                #
#---------------------------------------------------------------#
configured = 1
qDeep_path = '/home/project/scoreDml/resNet/QDeep2/tools/GDT-TS/'
if(configured == 0 or not os.path.exists(qDeep_path + '/apps/aleigen') or
           not os.path.exists(qDeep_path + '/apps/calNf_ly') or
           not os.path.exists(qDeep_path + '/apps/dssp') or
           not os.path.exists(qDeep_path + '/scripts/pdb2rr.pl') or
           not os.path.exists(qDeep_path + '/scripts/ros_energy.py')):
	print("\nError: not yet configured!\nPlease configure as follows\n$ cd QDeep\n$ python configure.py\n")
	exit(1)

#------------------------arguments------------------------------#
#Shows help to the users                                        #
#---------------------------------------------------------------#
parser = argparse.ArgumentParser()

parser._optionals.title = "Arguments"
parser.add_argument('--tgt', dest='target_name',
        default = '',    # default empty!
        help = 'Target name')

parser.add_argument('--seq', dest='seq_file',
        default = '',    # default empty!
        help = 'Sequence file')

parser.add_argument('--dcy', dest='decoy_dir',
        default = '',    # default empty!
        help = 'Decoy directory')

parser.add_argument('--aln', dest='aln_file',
        default = '',    # default empty!
        help = 'Multiple sequence alignment')

parser.add_argument('--dist', dest='distance_file',
        default = '',    # default empty!
        help = 'DMPfold predicted distance')

parser.add_argument('--pssm', dest='pssm_file',
        default = '',    # default empty!
        help = 'PSSM file')

parser.add_argument('--spd3', dest='spd33_file',
        default = '',    # default empty!
        help = 'SPIDER3 output (.spd33)')

parser.add_argument('--msa', dest='yes',
        default = 'no',    # default no!
        help = 'yes|no   Whether to use deep MSA (default: no)')

parser.add_argument('--gpu', dest='device_id',
        default = '-1',    # default cpu!
        help = 'Device id (0/1/2/3/4/..) Whether to run on GPU (default: CPU)')

parser.add_argument('--out', dest='output_path',
        default = '',    # default empty!
        help = 'Output directory name')

if len(sys.argv) < 8:
    parser.print_help(sys.stderr)
    sys.exit(1)

options = parser.parse_args()

seq_file = options.seq_file
target_name = options.target_name
decoy_dir = options.decoy_dir
aln_file = options.aln_file
pssm_file = options.pssm_file
spd3_file = options.spd33_file
dist_file = options.distance_file
deep_msa = options.yes
gpu = options.device_id
output_path = options.output_path

working_path = os.getcwd()

#----------------trained models and weights---------------------#
#                                                               #
#---------------------------------------------------------------#
if(deep_msa == 'no'):
        model_1_0 = qDeep_path + '/models/QDeep_standard/1.0/model_on_parameter_1'
        model_2_0 = qDeep_path + '/models/QDeep_standard/2.0/model_on_parameter_1'
        model_4_0 = qDeep_path + '/models/QDeep_standard/4.0/model_on_parameter_1'
        model_8_0 = qDeep_path + '/models/QDeep_standard/8.0/model_on_parameter_1'

        model_1_0_weight = qDeep_path + '/models/QDeep_standard/1.0/weights_on_parameter_1.h5'
        model_2_0_weight = qDeep_path + '/models/QDeep_standard/2.0/weights_on_parameter_1.h5'
        model_4_0_weight = qDeep_path + '/models/QDeep_standard/4.0/weights_on_parameter_1.h5'
        model_8_0_weight = qDeep_path + '/models/QDeep_standard/8.0/weights_on_parameter_1.h5'

if(deep_msa == 'yes'):
	model_1_0 = qDeep_path + '/models/QDeep_deep/1.0/model_on_parameter_1'
	model_2_0 = qDeep_path + '/models/QDeep_deep/2.0/model_on_parameter_1'
	model_4_0 = qDeep_path + '/models/QDeep_deep/4.0/model_on_parameter_1'
	model_8_0 = qDeep_path + '/models/QDeep_deep/8.0/model_on_parameter_1'

	model_1_0_weight = qDeep_path + '/models/QDeep_deep/1.0/weights_on_parameter_1.h5'
	model_2_0_weight = qDeep_path + '/models/QDeep_deep/2.0/weights_on_parameter_1.h5'
	model_4_0_weight = qDeep_path + '/models/QDeep_deep/4.0/weights_on_parameter_1.h5'
	model_8_0_weight = qDeep_path + '/models/QDeep_deep/8.0/weights_on_parameter_1.h5'

#------------------------sets GPU device------------------------#     
#                                                               #
#---------------------------------------------------------------#
if(gpu != ""):
	os.environ["CUDA_VISIBLE_DEVICES"] = gpu

#------------------------DO NOT change--------------------------#
#                                                               #
#---------------------------------------------------------------#
dssp_path = qDeep_path + '/apps/dssp'
stride_path = qDeep_path + '/apps/stride'
aleigen_path = qDeep_path + '/apps/aleigen'
neff_path = qDeep_path + '/apps/calNf_ly'
pdb2rr_path = qDeep_path + '/scripts/pdb2rr.pl'
ros_script = qDeep_path + '/scripts/ros_energy.py'

class QDeep():

        #------------------------constructor----------------------------#
        #Initialize necessary variables                                 #
        #---------------------------------------------------------------#
        def __init__(self, target):
                self.target_name = target
                self.seq_length = 0
                
                
        #------------------------check option---------------------------#
        #checks whether all args are passed                             #
        #---------------------------------------------------------------#
        def check_options(self):
                if (seq_file != "" and target_name != "" and decoy_dir != "" and 
                        aln_file != "" and pssm_file != "" and spd3_file != "" and 
                        dist_file != "" and output_path != ""):
                        return True
                else:
                        return False

        #---------------------validate input files----------------------#
        #checks validity of input file                                  #
        #---------------------------------------------------------------#
        def contains_number(self, str):
                return any(char.isdigit() for char in str)

        #-------validate sequence file-----------#
        #Invalid if the file does not exist
        #Invalid if the file is empty
        #Invalid if the sequence cotains any digit
        def validate_seq(self, seq_file):
                if(os.path.exists(seq_file)):
                        f = open(seq_file, 'r')
                        text = f.readlines()
                        text = [line.strip() for line in text if not '>' in line]
                        seqQ = ''.join( text )
                        self.seq_length = len(seqQ)
                        if(self.seq_length > 0):
                                return True
                        else:
                                return False
                else:
                        return False

        #-----------validate decoy dir-----------#
        #Invalid if dir not passed
        #Invalid if the dir does not exist
        #Invalid if the dir does not contain at least
        #1 pdb file with ATOM records
        #Invalid if all the pdb files are empty
        def validate_dec_dir(self, decoy_dir):
            valid = False
            global tot_decoy
            tot_decoy = 0
            if(os.path.isdir(decoy_dir)):
                    decoys = os.listdir(decoy_dir)
                    for i in range(len(decoys)):
                            #if(decoys[i].endswith('.pdb')):
                            dec_res_list=[]
                            dec_res_no = []
                            with open(decoy_dir + "/" + decoys[i]) as dFile:
                                    for line in dFile: 
                                            if(line[0:(0+4)]=="ATOM"):
                                                    dec_res_no.append(line[22:(22+4)])
                            dec_res_list=sorted((self.get_unique_list(dec_res_no)))
                            if(len(dec_res_list) > 0):
                                    tot_decoy += 1
            if(tot_decoy > 0):
                    valid = True
            return valid                
            
        #------------validate aln----------------#
        #Invalid if the file does not exist
        def validate_aln(self, aln_file):
                valid = False
                if(os.path.exists(aln_file)):
                        valid = True
                else:
                        valid = False
                return valid

        #--------validate distance file----------#
        #Invalid if the file does not exist
        def validate_dist(self, dist_file):
                valid = False
                if(os.path.exists(dist_file)):
                        valid = True
                else:
                        valid = False
                return valid
                        
        #----------validate PSSM file------------#
        #Invalid if the file does not exist
        #Invalid if the file is empty
        #Invalid if residue line < 43
        def validate_pssm(self, pssm_file):
                valid = False
                if(os.path.exists(pssm_file)):
                    with open(pssm_file) as fFile:
                            for line in fFile:
                                    tmp = line.split()
                                    if(len(tmp) > 0 and self.contains_number(tmp[0]) == True and
                                       len(tmp) < 42):
                                            valid = False
                                            break
                                    else:
                                            valid = True
                return valid

        #---------validate spd33 file-----------#
        #Invalid if the file does not exist
        #Invalid if the file is empty
        #Invalid if residue line < | > 13
        def validate_spd3(self, spd3_file):
                valid = False
                if(os.path.exists(spd3_file)):
                    with open(spd3_file) as fFile:
                            for line in fFile:
                                    tmp = line.split()
                                    if(len(tmp) > 0 and self.contains_number(tmp[0]) == True):
                                            if(len(tmp) > 13 or len(tmp) < 13):
                                                    valid = False
                                                    break
                                            
                                            else:
                                                    valid = True
                return valid
        

        #----------------------sigmoid----------------------------------#
        #purpose: scale numbers between 0 and 1                         #
        #parameter: number                                              #
        #---------------------------------------------------------------#
        def sigmoid(self, x):
              if x < 0:
                    return 1 - 1/(1 + math.exp(x))
              else:
                    return 1/(1 + math.exp(-x))

        #---------------------get_unique_list---------------------------#
        #purpose: takes a list and return a non-redundant list          #
        #parameter: list                                                #
        #---------------------------------------------------------------#
        def get_unique_list(self, in_list):
                if isinstance(in_list,list):
                        return list(set(in_list))

        #---------------------readFiles---------------------------------#
        #purpose: reads and store all files in a dir to an array        #
        #parameter: directory                                           #
        #---------------------------------------------------------------#
        def read_files(self, directory):
            global filesInDir
            filesInDir=[]
            for file in os.listdir(directory):
                    #getOnlyFileName=os.path.splitext(file.rsplit('.', 2)[0])[0]
                            #if(decoys[i].endswith('.pdb')):
                            dec_res_list=[]
                            dec_res_no = []
                            with open(decoy_dir + "/" + file) as dFile:
                                    for line in dFile: 
                                            if(line[0:(0+4)]=="ATOM"):
                                                    dec_res_no.append(line[22:(22+4)])
                            dec_res_list=sorted((self.get_unique_list(dec_res_no)))
                            if(len(dec_res_list) > 0):
                                    filesInDir.append(file)
            return filesInDir
                    
        #----------------------run_dssp or strid------------------------#
        #purpose: runs dssp or stride tool for generating SS and SA     #
        #if DSSP failes, STRIDE will run                                #
        #                                                               #
        #---------------------------------------------------------------#
        def run_dssp_stride(self):
                files = []
                files = self.read_files(decoy_dir)
                if not os.path.isdir(output_path+"/dssp"):
                        os.makedirs(output_path+"/dssp")

                if not os.path.isdir(output_path+"/stride"):
                        os.makedirs(output_path+"/stride")
                        
                for i in range(len(files)):
                        dssp_ret_code = os.system(dssp_path +" -i " + decoy_dir + "/" + files[i] + " -o " + 
                                output_path + "/dssp/" + os.path.splitext(files[i].rsplit('/', 1)[-1])[0] + ".dssp")

                        if(dssp_ret_code != 0):
                                print("DSSP failed to run. Running STRIDE for " + files[i])
                                os.system(stride_path +" " + decoy_dir + "/" + files[i] + ">" + 
                                        output_path + "/stride/" + os.path.splitext(files[i].rsplit('/', 1)[-1])[0] + ".stride")

        '''
        #-----------------------run_stride------------------------------#
        #purpose: runs stride tool for generating SS and SA             #
        #                                                               #
        #---------------------------------------------------------------#
        def run_stride(self):
                files = []
                files = self.read_files(decoy_dir)
                if not os.path.isdir(output_path+"/stride"):
                        os.makedirs(output_path+"/stride")
                for i in range(len(files)):
                        os.system(stride_path +" " + decoy_dir + "/" + files[i] + ">" + 
                                output_path + "/stride/" + os.path.splitext(files[i].rsplit('/', 1)[-1])[0] + ".stride")
        '''

        #-----------------------get_neff--------------------------------#
        #purpose: calculate NEFF                                        #
        #---------------------------------------------------------------#
        def get_neff(self):
                neff = 0
                with open(output_path + '/neff/' + self.target_name + '.neff') as n_file:
                        for line in n_file:
                                tmp = line.split()
                                if(len(tmp) > 0):
                                        x=np.array(tmp)
                                        x=np.asfarray(x, float)
                                        neff = sum(x) / len(x)

                return neff

        #------------------------get_cmo--------------------------------#
        #purpose: extract the CMO value                                 #
        #---------------------------------------------------------------#
        def get_cmo(self, cmo_file):
                cmo = 0
                count = 0
                with open(cmo_file) as n_file:
                        for line in n_file:
                                tmp = line.split()
                                if(len(tmp) > 0 and count == 1):
                                        cmo = float(tmp[0])
                                        break
                                count += 1
                return cmo

        #------------------------get8to3ss------------------------------#
        #purpose: converts 8 states SS to 3 states SS                   #
        #---------------------------------------------------------------#
        def get8to3ss(self, ss_parm):
                eTo3=""
                if (ss_parm == "H" or ss_parm == "G" or ss_parm == "I"):
                        eTo3="H"
                elif(ss_parm == "E" or ss_parm == "B"):
                        eTo3="E"
                else:
                        eTo3="C"
                return eTo3

        #------------------------get3to1aa------------------------------#
        #purpose: converts 3 states AA to 1 states AA                   #
        #---------------------------------------------------------------#
        def get3to1aa(self, aa):
            dict = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                 'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
                 'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
                 'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

            return dict[aa]

        #--------------------------get_rsa------------------------------#
        #purpose: extracts RSA                                          #
        #---------------------------------------------------------------#
        def get_rsa(self, amnAcidParam, saValParam):
                saVal = 0;
                aaSA=[];
                aaSaVal=[];
                aaSA=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'];
                aaSaVal=[115, 135, 150, 190, 210, 75, 195, 175, 200, 170, 185, 160, 145, 180, 225, 115, 140, 155, 255, 230];
                k=0
                while k<len(aaSA):
                        if(amnAcidParam==aaSA[k]):
                                saVal = float(saValParam) / aaSaVal[k]
                                break
                        else:
                                k+=1
                return saVal

        #----------------------gen_energy_terms-------------------------#
        #purpose: generate 12 Rosetta's energy terms                    #
        #---------------------------------------------------------------#
        def gen_energy_terms(self):
                files = []
                files = self.read_files(decoy_dir)
                if not os.path.isdir(output_path + "/rosetta"):
                        os.makedirs(output_path + "/rosetta")
                    
                ros_proc = subprocess.Popen('python -W ignore ' + ros_script + ' -d ' + decoy_dir +
                                  ' -o ' + output_path + "/rosetta > " + output_path + "/rosetta.log", shell=True).wait()
                if(ros_proc != 0):
                    print('Error occured while generating rosetta energy.\n' +
                              'Please check the installation')
                    exit()
                        
                                


        #---------------------generate_int_map--------------------------#
        #purpose: generate int map at diff thresholds                   #
        #---------------------------------------------------------------#
        def generate_int_map(self):
                if not os.path.isdir(output_path + "/int_maps"):
                        os.makedirs(output_path + "/int_maps")

                #------interaction map at threshold 6---------#
                #                                             #
                #---------------------------------------------#
                threshold = [6]
                for i in range(len(threshold)):
                        #os.chdir(output_path + "/int_maps")
                        outFile = open(output_path + "/int_maps/" + self.target_name + "_" + str(threshold[i]) + ".rr", "w")
                        with open(dist_file) as cFile:
                                for line in cFile:
                                        tmp = line.split()
                                        total_prob = 0
                                        string = ""
                                        for t in range(threshold[i] - 2):
                                                total_prob = total_prob + float(tmp[t + 2])
                                        outFile.write(tmp[0] + "  " + tmp[1] + "  " + str(total_prob) + "\n")
                        outFile.close()
                        #os.chdir(working_path)

                threshold = [8, 10, 12, 14]
                for i in range(len(threshold)):
                        #os.chdir(output_path + "/int_maps")
                        outFile = open(output_path + "/int_maps/" + self.target_name + "_" + str(threshold[i]) + ".rr", "w")
                        with open(dist_file) as cFile:
                                for line in cFile:
                                        tmp = line.split()
                                        total_prob = 0
                                        string = ""
                                        for t in range(threshold[i]):
                                                total_prob = total_prob + float(tmp[t + 2])
                                        outFile.write(tmp[0] + "  " + tmp[1] + "  " + str(total_prob) + "\n")
                        outFile.close() 


        #------------------------align_map------------------------------#
        #purpose: align predicted and true interaction map              #
        #---------------------------------------------------------------#
        def align_map(self):
                if not os.path.isdir(output_path + "/cmo"):
                        os.makedirs(output_path + "/cmo")
                
                decoys = os.listdir(decoy_dir)
                #------for 6A interact map------#
                #                               #
                #-------------------------------#
                if(os.path.exists(output_path + "/int_maps/" + self.target_name + '_6.rr')):
                        os.system('cp ' + output_path + "/int_maps/" + self.target_name + '_6.rr ' + output_path + "/cmo")
                os.chdir(output_path + "/cmo")
                #step3: format the predicted interaction map
                lines_pred = []
                with open(self.target_name + '_6.rr') as pCon:
                        for line in pCon:
                                lines_pred.append(line)

                out_file = open(self.target_name + '_6.rr', 'w')
                os.chdir(working_path)
                #open seq to get the number of residue#

                #write the length of the sequence
                #write only the first two column
                os.chdir(output_path + "/cmo")
                out_file.write(str(self.seq_length) + '\n')
                for l in range(len(lines_pred)):
                        temp_l = lines_pred[l].split()
                        #--------change int. map threshold here---------#
                        #                                               #
                        #-----------------------------------------------#
                        if(float(temp_l[2]) > 0.2):
                                out_file.write(str(int(temp_l[0]) -1) + ' ' + str(int(temp_l[1]) -1) + '\n')
                out_file.close()

                #step1: calculate int. map from the pdbs#
                #decoys = os.listdir(decoy_dir)
                for d in range(len(decoys)):
                        os.chdir(working_path)
                        os.system('perl ' + pdb2rr_path + ' ' + decoy_dir + '/' + decoys[d] + ' CB 5 8&> ' + output_path + "/cmo/" + decoys[d].split('.')[0] + '_true.rr')
                        os.chdir(output_path + "/cmo")
                        # step:2 format the rr file for aleigen
                        lines = []
                        with open(decoys[d].split('.')[0] + '_true.rr') as tcFile:
                                for line in tcFile:
                                        lines.append(line)
                                        

                        #open file
                        final_out = open(decoys[d].split('.')[0] + '_true.rr', 'w')
                        count = 0
                        for c in range(len(lines)):
                                temp_con = lines[c].split()
                                if(count == 0):
                                        final_out.write(str(len(temp_con[0])) + '\n')
                                else:
                                        final_out.write(str(int(temp_con[0]) - 1) + ' ' + str(int(temp_con[1]) - 1) + '\n')
                                count += 1

                        final_out.close()
                        #step4: run aleigen
                        os.system(aleigen_path + ' ' + decoys[d].split('.')[0] + '_true.rr' + ' '
                                + self.target_name + '_6.rr 6 &>' + decoys[d].split('.')[0] + '_6.cmo')
                os.chdir(working_path)

                #--------for 8A int. map--------#
                #                               #
                #-------------------------------#
                #copy the predicted int.map
                if(os.path.exists(output_path + "/int_maps/" + self.target_name + '_8.rr')):
                        os.system('cp ' + output_path + "/int_maps/" + self.target_name + '_8.rr ' + output_path + "/cmo")
                os.chdir(output_path + "/cmo")
                #step3: format the predicted int. map
                lines_pred = []
                with open(self.target_name + '_8.rr') as pCon:
                        for line in pCon:
                                lines_pred.append(line)

                outFile = open(self.target_name + '_8.rr', 'w')

                #open sequence file to get the number of residue#
                os.chdir(working_path)
                #write the length of the sequence
                #write only the first two column
                os.chdir(output_path + "/cmo")
                outFile.write(str(self.seq_length) + '\n')
                for l in range(len(lines_pred)):
                        temp_l = lines_pred[l].split()
                        #--------change int. map threshold here---------#
                        #						    #
                        #-----------------------------------------------#
                        if(float(temp_l[2]) > 0.2):
                                outFile.write(str(int(temp_l[0]) -1) + ' ' + str(int(temp_l[1]) -1) + '\n')
                outFile.close()

                #step1: calculate int. map from the pdbs#
                #decoys = os.listdir(decoy_dir)
                for d in range(len(decoys)):
                        os.chdir(working_path)
                        os.system('perl ' + pdb2rr_path + ' ' + decoy_dir + '/' + decoys[d] + ' CB 5 8&> ' + output_path + "/cmo/" + decoys[d].split('.')[0] + '_true.rr')
                        os.chdir(output_path + "/cmo")
                        # step:2 format the rr file for aleigen
                        lines = []
                        with open(decoys[d].split('.')[0] + '_true.rr') as tcFile:
                                for line in tcFile:
                                        lines.append(line)

                        #open file
                        final_out = open(decoys[d].split('.')[0] + '_true.rr', 'w')
                        count = 0
                        for c in range(len(lines)):
                                temp_con = lines[c].split()
                                if(count == 0):
                                        final_out.write(str(len(temp_con[0])) + '\n')
                                else:
                                        final_out.write(str(int(temp_con[0]) - 1) + ' ' + str(int(temp_con[1]) - 1) + '\n')
                                count += 1

                        final_out.close()
                        #step4: run aleigen
                        os.system(aleigen_path + ' ' + decoys[d].split('.')[0] + '_true.rr' + ' '
                              + self.target_name + '_8.rr 6 &>' + decoys[d].split('.')[0] + '_8.cmo')
                os.chdir(working_path)

                #--------for 10A int. map--------#
                #                                #
                #--------------------------------#
                #copy the predicted int. map
                if(os.path.exists(output_path + "/int_maps/" + self.target_name + '_10.rr')):
                        os.system('cp ' + output_path + "/int_maps/" + self.target_name + '_10.rr ' + output_path + "/cmo")
                os.chdir(output_path + "/cmo")
                #step3: format the predicted int. map
                lines_pred = []
                with open(self.target_name + '_10.rr') as p_con:
                        for line in p_con:
                                lines_pred.append(line)

                out_file = open(self.target_name + '_10.rr', 'w')

                #open seq to get the number of residue#
                os.chdir(working_path)
                #write the length of the sequence
                #write only the first two column
                os.chdir(output_path + "/cmo")
                out_file.write(str(self.seq_length) + '\n')
                for l in range(len(lines_pred)):
                        temp_l = lines_pred[l].split()
                        #--------change int. map threshold here---------#
                        #					            #
                        #-----------------------------------------------#
                        if(float(temp_l[2]) > 0.2):
                                out_file.write(str(int(temp_l[0]) -1) + ' ' + str(int(temp_l[1]) -1) + '\n')
                out_file.close()

                #step1: calculate int. map from the pdbs#
                #decoys = os.listdir(decoy_dir)
                for d in range(len(decoys)):
                        os.chdir(working_path)
                        os.system('perl ' + pdb2rr_path + ' ' + decoy_dir + '/' + decoys[d] + ' CB 5 8&> ' + output_path + "/cmo/" + decoys[d].split('.')[0] + '_true.rr')
                        os.chdir(output_path + "/cmo")

                        # step:2 format the rr file for aleigen
                        lines = []
                        with open(decoys[d].split('.')[0] + '_true.rr') as tcFile:
                                for line in tcFile:
                                        lines.append(line)

                        #open file
                        final_out = open(decoys[d].split('.')[0] + '_true.rr', 'w')
                        count = 0
                        for c in range(len(lines)):
                                temp_con = lines[c].split()
                                if(count == 0):
                                        final_out.write(str(len(temp_con[0])) + '\n')
                                else:
                                        final_out.write(str(int(temp_con[0]) - 1) + ' ' + str(int(temp_con[1]) - 1) + '\n')
                                count += 1

                        final_out.close()
                        #step4: run aleigen
                        os.system(aleigen_path + ' ' + decoys[d].split('.')[0] + '_true.rr' + ' '
                              + self.target_name + '_10.rr 6 &>' + decoys[d].split('.')[0] + '_10.cmo')

                os.chdir(working_path)
                #--------for 12A int. map--------#
                #                                #
                #--------------------------------#
                #copy the predicted int. map
                if(os.path.exists(output_path + "/int_maps/" + self.target_name + '_12.rr')):
                        os.system('cp ' + output_path + "/int_maps/" + self.target_name + '_12.rr ' + output_path + "/cmo")
                os.chdir(output_path + "/cmo")
                #step3: format the predicted int. map
                lines_pred = []
                with open(self.target_name + '_12.rr') as p_con:
                        for line in p_con:
                                lines_pred.append(line)

                out_file = open(self.target_name + '_12.rr', 'w')

                #open seq to get the number of residue#
                os.chdir(working_path)
                #write the length of the sequence
                #write only the first two column
                os.chdir(output_path + "/cmo")
                out_file.write(str(self.seq_length) + '\n')
                for l in range(len(lines_pred)):
                        temp_l = lines_pred[l].split()
                        #--------change int. map threshold here---------#
                        #						    #
                        #-----------------------------------------------#
                        if(float(temp_l[2]) > 0.2):
                                out_file.write(str(int(temp_l[0]) -1) + ' ' + str(int(temp_l[1]) -1) + '\n')
                out_file.close()

                #step1: calculate int. maps from the pdbs#
                #decoys = os.listdir(decoy_dir)
                for d in range(len(decoys)):
                        os.chdir(working_path)
                        os.system('perl ' + pdb2rr_path + ' ' + decoy_dir + '/' + decoys[d] + ' CB 5 8&> ' + output_path + "/cmo/" + decoys[d].split('.')[0] + '_true.rr')
                        os.chdir(output_path + "/cmo")

                        # step:2 format the rr file for aleigen
                        lines = []
                        with open(decoys[d].split('.')[0] + '_true.rr') as tc_file:
                                for line in tc_file:
                                        lines.append(line)

                        #open file
                        final_out = open(decoys[d].split('.')[0] + '_true.rr', 'w')
                        count = 0
                        for c in range(len(lines)):
                                temp_con = lines[c].split()
                                if(count == 0):
                                        final_out.write(str(len(temp_con[0])) + '\n')
                                else:
                                        final_out.write(str(int(temp_con[0]) - 1) + ' ' + str(int(temp_con[1]) - 1) + '\n')
                                count += 1

                        final_out.close()
                        #step4: run aleigen
                        os.system(aleigen_path + ' ' + decoys[d].split('.')[0] + '_true.rr' + ' '
                              + self.target_name + '_12.rr 6 &>' + decoys[d].split('.')[0] + '_12.cmo')


                os.chdir(working_path)

                #--------for 14A int. map--------#
                #                                #
                #--------------------------------#
                #copy the predicted int. map
                if(os.path.exists(output_path + "/int_maps/" + self.target_name + '_14.rr')):
                        os.system('cp ' + output_path + "/int_maps/" + self.target_name + '_14.rr ' + output_path + "/cmo")
                os.chdir(output_path + "/cmo")
                lines_pred = []
                with open(self.target_name + '_14.rr') as p_con:
                        for line in p_con:
                                lines_pred.append(line)

                out_file = open(self.target_name + '_14.rr', 'w')

                #open seq to get the number of residue#
                os.chdir(working_path)
                #write the length of the sequence
                #write only the first two column
                os.chdir(output_path + "/cmo")
                out_file.write(str(self.seq_length) + '\n')
                for l in range(len(lines_pred)):
                        temp_l = lines_pred[l].split()
                        #--------change int_map threshold here----------#
                        #                                               #
                        #-----------------------------------------------#
                        if(float(temp_l[2]) > 0.2):
                                out_file.write(str(int(temp_l[0]) -1) + ' ' + str(int(temp_l[1]) -1) + '\n')
                out_file.close()

                #step1: calculate int_map from the pdbs#
                #decoys = os.listdir(decoy_dir)
                for d in range(len(decoys)):
                        os.chdir(working_path)
                        os.system('perl ' + pdb2rr_path + ' ' + decoy_dir + '/' + decoys[d] + ' CB 5 8&> ' + output_path + "/cmo/" + decoys[d].split('.')[0] + '_true.rr')
                        os.chdir(output_path + "/cmo")

                        # step:2 format the rr file for aleigen
                        lines = []
                        with open(decoys[d].split('.')[0] + '_true.rr') as tc_file:
                                for line in tc_file:
                                        lines.append(line)

                        #open file
                        final_out = open(decoys[d].split('.')[0] + '_true.rr', 'w')
                        count = 0
                        for c in range(len(lines)):
                                temp_con = lines[c].split()
                                if(count == 0):
                                        final_out.write(str(len(temp_con[0])) + '\n')
                                else:
                                        final_out.write(str(int(temp_con[0]) - 1) + ' ' + str(int(temp_con[1]) - 1) + '\n')
                                count += 1

                        final_out.close()
                        #step4: run aleigen
                        os.system(aleigen_path + ' ' + decoys[d].split('.')[0] + '_true.rr' + ' '
                              + self.target_name + '_14.rr 6 &>' + decoys[d].split('.')[0] + '_14.cmo')

                os.chdir(working_path)       

        #------run and process neff-------#
        #input: aln                       #
        #---------------------------------#
        def generate_neff(self):
            if not os.path.isdir(output_path + "/neff"):
                    os.makedirs(output_path + "/neff")
            os.system('cp ' + aln_file + ' ' + output_path + "/neff")
            os.chdir(output_path + "/neff")
            os.system(neff_path + ' ' + self.target_name + '.aln 0.8&> ' + self.target_name + '.neff')
            os.chdir(working_path)

        #---------------------------process sliding window if used during training-------------------------#
        #                                                                                                  #
        #--------------------------------------------------------------------------------------------------#
        def processSlidingWindow_train_with_0(self, featureFile, targetDir, window_size):
                del features[:]
                del finalFeatures[:]
                ##reading all features to a list##    
                with open(featureFile) as fFile:
                        first_res = 1
                        for line in fFile:
                                #pad features for blanks#
                                #---pad feat at the beginning---#
                                #                               #
                                #-------------------------------#
                                if(first_res == 1):
                                        feat_len = len(line.split())
                                        #how many lines to pad? = (window_size-1)/2
                                        for p in range(int((window_size - 1)/2)):
                                                feat = ""
                                                #how many features? = len of feat
                                                for l in range(feat_len):
                                                        feat += "0 "
                                                features.append(feat)
                                features.append(line.rstrip())
                                first_res = 0
                        #------pad feat at the end------#
                        #                               #
                        #-------------------------------#
                        feat_len = len(line.split())
                        #how many lines to pad? = window_size/2 - 1
                        for p in range(int((window_size - 1)/2)):
                                feat = ""
                                #how many features? = len of feat + 1 (label)
                                for l in range(feat_len):
                                        feat += "0 "
                                features.append(feat)
                        
                start = 0
                for i in range((len(features)-int(window_size))+1):
                        tmpFeatures=[]
                        tmpIndFtrs=[]
                        window_label=[]
                        label=0
                        ##label position (middle)##
                        window_label=(int(window_size)/2)+1
                        ##read till total no of window size##
                        for j in range (int(window_size)):
                                ##taking all features except the label##
                                tmp=features[j+start].split()
                                tmpFeatures.append(tmp)
                                ##determining the label (middle)##
                                #if(j+1==int(window_label)):
                                #    label=tmp[-1]
                        start=start+1
                        ##store individual features for concatenation##
                        for k in range(len(tmpFeatures)):
                                for l in range(len(tmpFeatures[k])):
                                        tmpIndFtrs.append(tmpFeatures[k][l])
                        len(tmpIndFtrs)
                        finalFeatures.append(" ".join(tmpIndFtrs))

                #-----write features------#
                #                         #
                #-------------------------#
                featOut = open(targetDir + "/" + os.path.splitext(featureFile.rsplit('/', 1)[-1])[0] + ".window_feat", "w")
                for feat in range(len(finalFeatures)):
                        featOut.write(finalFeatures[feat] + "\n")
                featOut.close()


        #-----------------------------------Step 1:process unique features---------------------------------------------#
        #                                                                                                              #
        #--------------------------------------------------------------------------------------------------------------#
        def generate_feature(self):
                #----------get neff feat---------#
                #                                #
                #--------------------------------#
                noOfEffSeq = self.get_neff()

                #-------Failed decoy log---------#
                #                                #
                #--------------------------------#
                global total_failed_decoy
                total_failed_decoy = 0
                failed_decoy = open(output_path + '/failed_decoy.log', 'w')
                
                #check here whether all features have been generated or not#
                #fetch lines from files those are common#
                pssmLines = []
                spd3Lines = []

                if os.path.isfile(pssm_file):
                    with open(pssm_file) as fFile:
                            for line in fFile:                    
                                    tmp = line.split()
                                    if(len(tmp) > 0):
                                            pssmLines.append(line)
                                
                if os.path.isfile(spd3_file):
                    with open(spd3_file) as fFile:
                            for line in fFile:                    
                                    tmp = line.split()
                                    if(len(tmp) > 0 and tmp[0] != "#"):
                                            spd3Lines.append(line)


                if not os.path.isdir(output_path + "/features"):
                        os.makedirs(output_path + "/features")

                if not os.path.isdir(output_path + "/residue_list"):
                        os.makedirs(output_path + "/residue_list")
                #---------------for each decoy---------------#
                #                                            #
                #--------------------------------------------#
                #read all deocoys for the taraget
                self.read_files(decoy_dir)
                for d in range(len(filesInDir)): #for each decoy
                    outputFeat = open(output_path + "/features/" + filesInDir[d].split('.')[0] + ".feat", "w")

                    #--------get CMO--------#
                    #                       #
                    #-----------------------#
                    cmo_score_6 = 0
                    cmo_score_6 = float(self.get_cmo(output_path + '/cmo/' + filesInDir[d].split('.')[0] + '_6.cmo') * 10) / float(100)

                    cmo_score_8 = 0
                    cmo_score_8 = float(self.get_cmo(output_path + '/cmo/' + filesInDir[d].split('.')[0] + '_8.cmo') * 25) / float(100)
                    
                    cmo_score_10 = 0
                    cmo_score_10 = float(self.get_cmo(output_path + '/cmo/' + filesInDir[d].split('.')[0] + '_10.cmo') * 30) / float(100)
                    
                    cmo_score_12 = 0
                    cmo_score_12 = float(self.get_cmo(output_path + '/cmo/' + filesInDir[d].split('.')[0] + '_12.cmo') * 25) / float(100)

                    cmo_score_14 = 0
                    cmo_score_14 = float(self.get_cmo(output_path + '/cmo/' + filesInDir[d].split('.')[0] + '_14.cmo') * 10) / float(100)
                    
                    #get residue list from the reference model
                    ######################################get residue index#################################################
                    residueList=[]
                    tmp_residue_list = []
                    start_end_ResNo = []                                                                                  ##
                    with open(decoy_dir + "/" + filesInDir[d]) as file:                                                   ##
                        for line in file:                                                                                 ##
                            if(line[0:(0+4)]=="ATOM"):                                                                    ##
                                start_end_ResNo.append(line[22:(22+4)])                                                   ##
                                                                                                                          ##
                    residueList=sorted((self.get_unique_list(start_end_ResNo)))                                           ##
                    residueList=list(map(int, residueList))                                                               ##
                    ######################################process each decoy file with respect to the native################
                    #-------residue-wise feature extraction------#
                    #                                            #
                    #--------------------------------------------#

                    #--------for each residue of a decoy---------#
                    #                                            #
                    #--------------------------------------------#
                    total_phi = 0
                    total_psi = 0
                    total_feat_res = 0
                    angular_rmsd_phi = ""
                    angular_rmsd_psi = ""
                    angular_rmsd_norm_phi = ""
                    angular_rmsd_norm_psi = ""
                    for r in range(len(residueList)):
                        feat_pssm = ""
                        feat_ss = ""
                        feat_sa = 0
                        feat_rosetta = ""
                        feat_mass = ""
                        feat_rosetta = ""
                        lga_label = ""
                        feat_max_prob = 0
                        
                        #----------PSSM features--------#
                        #                               #
                        #-------------------------------#
                        for p in range(len(pssmLines)):                 
                            tmp = pssmLines[p].split()
                            if(len(tmp) > 20 and tmp[0] != 'A' and tmp[0] != 'K' and tmp[0] != 'Standard' and
                               tmp[0] != 'PSI' and tmp[0] != 'Last' and int(tmp[0]) == residueList[r]):
                                feat_pssm += str(self.sigmoid(float(tmp[43])))
                                break
                            
                        #-------SA and SS features------#
                        #                               #
                        #-------------------------------#
                        #SS and SA features (for each residue, after matching with the residue, extract ss and sa)
                        ss=""
                        sa=""
                        aa=""

                        
                        #-------Process each DSSP-------#
                        #                               #
                        #-------------------------------#
                        dsspResFound = 0
                        if os.path.isfile(output_path + "/dssp/" + filesInDir[d].split('.')[0] + ".dssp"):
                            with open(output_path + "/dssp/" + filesInDir[d].split('.')[0] + ".dssp") as fp:
                                line = fp.readline()
                                cntLn = 0 ##line counter##
                                
                                residue="#"
                                while line: ##reading each line##
                                    line = fp.readline()
                                    if (cntLn<1):
                                        if (line[2:(2+len(residue))] == residue):
                                            cntLn+=1
                                            continue
                                        
                                    if (cntLn>0):
                                        if (len(line)>0):
                                            if(line[13:(13+1)].isalpha()):
                                                ssSeq = line[16:(16+1)]
                                                saSeq = line[35:(35+3)].strip()
                                                aaSeqNo = line[6:(6+4)].strip()
                                                aaSeq = line[13:(13+1)]
                                                dsspPhi = float(line[103:(103+6)])
                                                dsspPsi = float(line[109:(109+6)])
                                                ##8 to 3 state conversion
                                                #ss.append(get8to3ss(ssSeq))
                                                #if the residue numbers match
                                                if(int(aaSeqNo) == residueList[r]):
                                                    ss = self.get8to3ss(ssSeq)
                                                    sa = saSeq
                                                    aa = aaSeq
                                                    dsspResFound = 1
                                                    break
                        

                        else:
                        #------Process each stride------#
                        #                               #
                        #-------------------------------#
                                dsspResFound = 0
                                if os.path.isfile(output_path + "/stride/" + filesInDir[d].split('.')[0] + ".stride"):
                                    with open(output_path + "/stride/" + filesInDir[d].split('.')[0] + ".stride") as fp:
                                        line = fp.readline()
                                        cntLn = 0 ##line counter##
                                        
                                        residue="Residue"
                                        while line: ##reading each line##
                                                line = fp.readline()
                                                if (len(line)>0):
                                                    if((line[0:(0+3)]) == "ASG"):
                                                        ssSeq = line[24:(24+1)]
                                                        saSeq = line[64:(64+5)].strip()
                                                        aaSeqNo = line[11:(11+4)].strip()
                                                        aaSeq = self.get3to1aa(line[5:(5+3)].strip())
                                                        dsspPhi = float(line[42:(43+7)])
                                                        dsspPsi = float(line[52:(53+7)])
                                                        ##8 to 3 state conversion
                                                        #ss.append(get8to3ss(ssSeq))
                                                        #if the residue numbers match
                                                        if(int(aaSeqNo) == residueList[r]):
                                                            ss = self.get8to3ss(ssSeq)
                                                            sa = saSeq
                                                            aa = aaSeq
                                                            dsspResFound = 1
                                                            break
                                                
                        #agreement between spd3 and dssp (parse dssp file)
                        spdSS = ""
                        spdSA = ""
                        spdAA = ""
                        spdResFound = 0
                        for spd in range(len(spd3Lines)):
                            spdSeqNo = spd3Lines[spd][0:(0+3)]
                            #if the residue numbers match
                            if(int(spdSeqNo) == residueList[r]):
                                spdSS = spd3Lines[spd][6:(6+1)]
                                spdSA = spd3Lines[spd][8:(8+5)]
                                spdAA = spd3Lines[spd][4:(4+1)]
                                spdPhi = float(spd3Lines[spd][14:(14+6)])
                                spdPsi = float(spd3Lines[spd][21:(21+6)])
                                feat_max_prob = max(float(spd3Lines[spd][52:(52+5)]), float(spd3Lines[spd][58:(58+5)]), float(spd3Lines[spd][64:(64+5)]))
                                spdResFound = 1
                                break

                        if(dsspResFound == 1 and spdResFound == 1):
                            if(spdSS == ss):
                                feat_ss = "1"
                            else:
                                feat_ss = "0"

                            #squared error
                            feat_sa = pow(self.get_rsa(spdAA, spdSA) - self.get_rsa(aa, sa), 2)
                            #------ROSETTA energy terms-----#
                            #                               #
                            #-------------------------------#
                            if os.path.isfile(output_path + "/rosetta/" + filesInDir[d].split('.')[0] + ".rosetta"):
                                with open(output_path + "/rosetta/" + filesInDir[d].split('.')[0] + ".rosetta") as rosFile:
                                    for line in rosFile:
                                        tmpRos = line.split()
                                        if(len(tmpRos) > 0):
                                            if(int(tmpRos[0]) == residueList[r]):
                                                for ros in range(1, 13):
                                                    feat_rosetta += str(self.sigmoid(float(tmpRos[ros]))) + " "
                                                break        

                        comp_feature = str(feat_pssm) + " " + str(feat_ss) + " " + str(feat_sa) + " " + str(feat_rosetta)

                        #----for testing purpose------#
                        #print(str(residueList[r]) + " " + targets[i] + "/" + filesInDir[d].split('.')[0])
                        #print(comp_feature)
                        #-----------------------------#
                        if(len(comp_feature.split()) == 15):
                            outputFeat.write(comp_feature + "\n")
                            
                            tmp_residue_list.append(residueList[r])

                            #---------angular RMSD--------#
                            #                             #  
                            #-----------------------------#
                            del_phi1 = abs(dsspPhi - spdPhi)
                            del_phi2 = (2 * math.pi) - (abs(dsspPhi - spdPhi))
                            del_phi_min = math.pow(min(del_phi1, del_phi2), 2)
                            total_phi = total_phi + del_phi_min

                            del_psi1 = abs(dsspPsi - spdPsi)
                            del_psi2 = (2 * math.pi) - (abs(dsspPsi - spdPsi))
                            del_psi_min = math.pow(min(del_psi1, del_psi2), 2)
                            total_psi = total_psi + del_psi_min
                            total_feat_res += 1
                            #-----------------------------#
                        
                    if(total_feat_res > 0):
                        angular_rmsd_phi = float(math.sqrt(total_phi / total_feat_res))
                        angular_rmsd_psi = float(math.sqrt(total_psi / total_feat_res))

                        angular_rmsd_norm_phi = 1 / (1 + math.pow(((angular_rmsd_phi / (math.pi / 4))), 2))
                        angular_rmsd_norm_psi = 1 / (1 + math.pow(((angular_rmsd_psi / (math.pi / 4))), 2))
                    outputFeat.close()

                    #--------open file to add angular rmsd feat------------#
                    #                                                      #
                    #------------------------------------------------------#
                    with open(output_path + "/features/" + filesInDir[d].split('.')[0] + ".feat") as featFile:
                        lines = []
                        for line in featFile:
                            tmp = line.split()
                            if(len(tmp) > 0):
                                lines.append(line)

                    outputFeat = open(output_path + "/features/" + filesInDir[d].split('.')[0] + ".feat", "w")
                    for af in range(len(lines)):
                        temp_l = lines[af].split()
                        finalFeat = str(temp_l[0] + " " + temp_l[1] + " " + temp_l[2] + " " + temp_l[3] + " " + temp_l[4] + " " + temp_l[5] + " " + temp_l[6] + " "
                        + temp_l[7] + " " + temp_l[8] + " " + temp_l[9] + " " + temp_l[10] + " " + temp_l[11] + " " + temp_l[12] + " " + temp_l[13] + " "
                        + temp_l[14] + " " +  str(angular_rmsd_norm_phi) + " " + str(angular_rmsd_norm_psi) + " " + str(cmo_score_6) + " " + str(cmo_score_8) + " " 
                        + str(cmo_score_10) + " " + str(cmo_score_12) + " " + str(cmo_score_14) + " " +  str(noOfEffSeq))

                        if(len(finalFeat.split()) == 23):

                            outputFeat.write(finalFeat + "\n")
                    outputFeat.close()

                    residue_list_out = open(output_path + "/residue_list/" + filesInDir[d].split('.')[0] + ".resList", 'w')
                    for res in range(len(tmp_residue_list)):
                        residue_list_out.write(str(tmp_residue_list[res]) + '\n')

                    #error checking: if the file is empty, then remove it#
                    if(os.stat(output_path + "/features/" + filesInDir[d].split('.')[0] + ".feat").st_size == 0):
                        print(output_path + "/features/" + filesInDir[d].split('.')[0] + ".feat" + " is empty. Removing...")
                        os.system("rm " + output_path + "/features/" + filesInDir[d].split('.')[0] + ".feat")
                        
                        #writing failed decoy log#
                        total_failed_decoy += 1
                        failed_decoy.write(filesInDir[d] + '\n')

        #----------------------------------Step 2:load and save models-------------------------------------------------#
        #                                                                                                              #
        #--------------------------------------------------------------------------------------------------------------#
        def score(self):
                print("Loading models...")

                #---------1.0---------#
                #                     #
                #---------------------#
                json_file = open(model_1_0, 'r')
                saved_model_json = json_file.read()
                json_file.close()
                saved_model_1_0 = model_from_json(saved_model_json)
                saved_model_1_0.load_weights(model_1_0_weight) #.h5 format
                saved_model_1_0.compile(optimizer='adam', loss='binary_crossentropy')

                #---------2.0---------#
                #                     #
                #---------------------#
                json_file = open(model_2_0, 'r')
                saved_model_json = json_file.read()
                json_file.close()
                saved_model_2_0 = model_from_json(saved_model_json)
                saved_model_2_0.load_weights(model_2_0_weight) #.h5 format
                saved_model_2_0.compile(optimizer='adam', loss='binary_crossentropy')
                #---------4.0---------#
                #                     #
                #---------------------#
                json_file = open(model_4_0, 'r')
                saved_model_json = json_file.read()
                json_file.close()
                saved_model_4_0 = model_from_json(saved_model_json)
                saved_model_4_0.load_weights(model_4_0_weight) #.h5 format
                saved_model_4_0.compile(optimizer='adam', loss='binary_crossentropy')

                #---------8.0---------#
                #                     #
                #---------------------#
                json_file = open(model_8_0, 'r')
                saved_model_json = json_file.read()
                json_file.close()
                saved_model_8_0 = model_from_json(saved_model_json)
                saved_model_8_0.load_weights(model_8_0_weight) #.h5 format
                saved_model_8_0.compile(optimizer='adam', loss='binary_crossentropy')


                #----------------------Step 3:process sliding window for each feature file and predict-------------------------#
                #                                                                                                              #
                #--------------------------------------------------------------------------------------------------------------#
                if not os.path.isdir(output_path + "/prediction"):
                    os.makedirs(output_path + "/prediction")
                score_file = open(output_path + "/" + self.target_name + ".QDeep", "w")
                for file in os.listdir(output_path + "/features/"):
                    if(file.endswith(".feat") and os.stat(output_path + "/features/" + file).st_size != 0):
                        
                        self.processSlidingWindow_train_with_0(output_path + "/features/" + file, output_path + "/features/", 21)
                        #-------------------------predict-------------------------#
                        #                                                         #
                        #---------------------------------------------------------#
                        pred_1_0 = open(output_path + "/prediction/" + os.path.splitext(file.rsplit('/', 1)[-1])[0] + "_pred_1_0.txt", "w")
                        pred_2_0 = open(output_path + "/prediction/" + os.path.splitext(file.rsplit('/', 1)[-1])[0] + "_pred_2_0.txt", "w")
                        pred_4_0 = open(output_path + "/prediction/" + os.path.splitext(file.rsplit('/', 1)[-1])[0] + "_pred_4_0.txt", "w")
                        pred_8_0 = open(output_path + "/prediction/" + os.path.splitext(file.rsplit('/', 1)[-1])[0] + "_pred_8_0.txt", "w")


                        residue_list_pdb = []
                        residue_list_pdb = open(output_path + "/residue_list/" + file.split('.')[0] + '.resList','r').readlines()
                        
                        for m_feat in range(len(finalFeatures)):
                            #format features#
                            x = np.array([float(i) for i in finalFeatures[m_feat].split()])
                            x = x.reshape((1, len(finalFeatures[m_feat].split()), 1))
                            pred_1_0.write(str(residue_list_pdb[m_feat].strip()) + '   ' + str(float(saved_model_1_0.predict(x)[0])) + "\n")
                            pred_2_0.write(str(residue_list_pdb[m_feat].strip()) + '   ' + str(float(saved_model_2_0.predict(x)[0])) + "\n")
                            pred_4_0.write(str(residue_list_pdb[m_feat].strip()) + '   ' + str(float(saved_model_4_0.predict(x)[0])) + "\n")
                            pred_8_0.write(str(residue_list_pdb[m_feat].strip()) + '   ' + str(float(saved_model_8_0.predict(x)[0])) + "\n")
                        
                        pred_1_0.close()
                        pred_2_0.close()
                        pred_4_0.close()
                        pred_8_0.close()
                        count_1_0 = 0
                        count_2_0 = 0
                        count_4_0 = 0
                        count_8_0 = 0

                        with open(output_path + "/prediction/" + os.path.splitext(file.rsplit('/', 1)[-1])[0] + "_pred_1_0.txt") as fFile:
                            for line in fFile:
                                if(float(line.split()[1]) > 0.5):
                                    count_1_0 += 1
                        with open(output_path + "/prediction/" + os.path.splitext(file.rsplit('/', 1)[-1])[0] + "_pred_2_0.txt") as fFile:
                            for line in fFile:
                                if(float(line.split()[1]) > 0.5):
                                    count_2_0 += 1
                        with open(output_path + "/prediction/" + os.path.splitext(file.rsplit('/', 1)[-1])[0] + "_pred_4_0.txt") as fFile:
                            for line in fFile:
                                if(float(line.split()[1]) > 0.5):
                                    count_4_0 += 1
                        with open(output_path + "/prediction/" + os.path.splitext(file.rsplit('/', 1)[-1])[0] + "_pred_8_0.txt") as fFile:
                            for line in fFile:
                                if(float(line.split()[1]) > 0.5):
                                    count_8_0 += 1

                        #------calculate GDT-TS------#
                        #                            #
                        #----------------------------#
                        frac_gdt_ts_1_0 = float(count_1_0)/float(len(finalFeatures))
                        frac_gdt_ts_2_0 = float(count_2_0)/float(len(finalFeatures))
                        frac_gdt_ts_4_0 = float(count_4_0)/float(len(finalFeatures))
                        frac_gdt_ts_8_0 = float(count_8_0)/float(len(finalFeatures))

                        gdt_ts_pred = (frac_gdt_ts_1_0 + frac_gdt_ts_2_0 + frac_gdt_ts_4_0 + frac_gdt_ts_8_0) / 4.0
                        print(file.split('.')[0] + ": " + str(gdt_ts_pred))
                        score_file.write(os.path.splitext(file.rsplit('.', 2)[0])[0] + " " + str(gdt_ts_pred) +"\n")
                score_file.close()
        #----------------------Sort score file--------------------------#
        #purpose: takes the score file and sort in desc order           #
        #---------------------------------------------------------------#
        def sort_scores(self):
                #sort the score file#
                with open(output_path + "/" + target_name + ".QDeep") as sFile:
                    lines = []
                    for line in sFile:
                            tmp = line.split()
                            if(len(tmp) > 0):
                                    lines.append(line)

                score_file = open(output_path + "/" + target_name + ".QDeep", 'w')
                for line in sorted(lines, key=lambda line: float(line.split()[1]), reverse = True):
                    score_file.write(line)
                score_file.close()

        #----------------------add failed decoy-------------------------#
        #purpose: add failed decoy to the score file if any             #
        #---------------------------------------------------------------#
        def add_failed_decoy(self):
                #sort the score file#
                with open(output_path + "/" + target_name + ".QDeep") as sFile:
                    lines = []
                    for line in sFile:
                            tmp = line.split()
                            if(len(tmp) > 0):
                                    lines.append(line)

                with open(output_path + "/" + "failed_decoy.log") as fFile:
                    for line in fFile:
                            if(len(tmp) > 0):
                                    lines.append(line.rstrip() + ' X\n')

                score_file = open(output_path + "/" + target_name + ".QDeep", 'w')
                for s in range(len(lines)):
                    score_file.write(lines[s])
                score_file.close()
                        
                                        
                                        


def main():
        #Create an instance of the class#
        q = QDeep(target_name)
        if(q.check_options() == True):

                print('Processing started for: ' + target_name)
                print('\n#-----------Validating input files-----------#\n' +
                      '#                                            #\n' +
                      '#--------------------------------------------#')
                
                if(q.validate_seq(seq_file) == True):
                        print('Checking sequence file: OK')
                else:
                        print('Error: Please check the sequence file')
                        exit()
                        
                if(q.validate_aln(aln_file) == True):
                        print('Checking aln file: OK')
                else:
                        print('Error: Please check the aln file')
                        exit()
                        
                if(q.validate_dist(dist_file) == True):
                        print('Checking distance file: OK')
                else:
                        print('Error: Please check the distance file')
                        exit()

                if(q.validate_pssm(pssm_file) == True):
                        print('Checking PSSM file: OK')
                else:
                        print('Error: Please check the PSSM file')
                        exit()

                if(q.validate_spd3(spd3_file) == True):
                        print('Checking SPIDER3 output file: OK')
                else:
                        print('Error: Please check the SPIDER3 output file')
                        exit()

                if(q.validate_dec_dir(decoy_dir) == True):
                        print('Total pdb file(s) accepted: ' + str(tot_decoy) + '\n')
                else:
                        print('Error: Please check the decoy directory')
                        exit()

                print('\n#------------Generating features-------------#\n' +
                      '#                                            #\n' +
                      '#--------------------------------------------#')
                if not os.path.isdir(output_path):
                        os.makedirs(output_path)

                print('Processing SS and SA...')
                q.run_dssp_stride()
                print('DONE!\n')
                print('Calculating energy terms...')
                q.gen_energy_terms()
                print('DONE!\n')
                print('Calculating NEFF...')
                q.generate_neff()
                print('DONE!\n')
                print('Processing distance features...')
                q.generate_int_map()
                q.align_map()
                print('DONE!\n')
                print('Finalizing feature generation...')
                q.generate_feature()
                print('Total failed decoy: ' + str(total_failed_decoy))
                if(total_failed_decoy > 0):
                        print('see failed decoy log: ' + output_path + '/failed_decoy.log')
                print('Total file(s) to be scored: ' + str(tot_decoy - total_failed_decoy))
                print('DONE!\n')
                print('\n#-----------------Scoring--------------------#\n' +
                      '#                                            #\n' +
                      '#--------------------------------------------#')
                q.score()
                q.sort_scores()
                if(total_failed_decoy > 0):
                        q.add_failed_decoy()
                print('\nCongratulations! All process are successfully done!')
                end_time = time.monotonic()
                total_time = timedelta(seconds=end_time - start_time)
                print('Total processing time: ' + str(total_time))
                print('See QDeep output: ' + output_path + '/' + target_name + '.QDeep\n')
    
        else:
            print("Please check your command")
            
if __name__ == '__main__':
        main()
