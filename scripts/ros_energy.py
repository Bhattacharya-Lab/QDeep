#!/usr/bin/python
#######################################################################################################
# Name: ros_ene.py
# Purpose: Protein single-model quality assessment using Residual Neural Network
# Developed by: Debswapna Bhattacharya
# Developed on: 1/17/2020
# Modified by: Md Hossain Shuvo
# Change log:
#######################################################################################################
import os, sys, re, random, math
import numpy as np
import optparse
from pyrosetta import *
from rosetta import *
from rosetta.core.scoring import *
init(extra_options = "-ignore_zero_occupancy false -ignore_unrecognized_res true -chemical:exclude_patches LowerDNA  UpperDNA Cterm_amidation SpecialRotamer VirtualBB ShoveBB VirtualDNAPhosphate VirtualNTerm CTermConnect sc_orbitals pro_hydroxylated_case1 pro_hydroxylated_case2 ser_phosphorylated thr_phosphorylated  tyr_phosphorylated tyr_sulfated lys_dimethylated lys_monomethylated  lys_trimethylated lys_acetylated glu_carboxylated cys_acetylated tyr_diiodinated N_acetylated C_methylamidated MethylatedProteinCterm")
parser = optparse.OptionParser()

parser.add_option('-d', dest='decoy_dir',
        default = '',    # default empty!
        help = 'decoy_dir')

parser.add_option('-o', dest='out_path',
        default = '',    # default empty!
        help = 'output_path')


(options,args) = parser.parse_args()
decoy_dir = options.decoy_dir
output_path = options.out_path

#---------------------get_unique_list---------------------------#
#purpose: takes a list and return a non-redundant list          #
#parameter: list                                                #
#---------------------------------------------------------------#
def get_unique_list(in_list):
        if isinstance(in_list,list):
            return list(set(in_list))

decoys = os.listdir(decoy_dir)

for i in range(len(decoys)):
        dec_res_list=[]
        dec_res_no = []
        with open(decoy_dir + "/" + decoys[i]) as dFile:
                for line in dFile: 
                        if(line[0:(0+4)]=="ATOM"):
                                dec_res_no.append(line[22:(22+4)])
        dec_res_list=sorted((get_unique_list(dec_res_no)))
        if(len(dec_res_list) > 0):
        #if(decoys[i].endswith('.pdb')):
                residueList=[]        
                start_end_ResNo = []    
                with open(decoy_dir + '/' + decoys[i]) as file:        
                        for line in file:                                         
                                if(line[0:(0+4)]=="ATOM"):                                                       
                                    start_end_ResNo.append(line[22:(22+4)])
                residueList=sorted((get_unique_list(start_end_ResNo)))                                                
                residueList=list(map(int, residueList))                       
                try:
                        pose = Pose()
                        pdb = Pose()
                        pose_from_file(pdb, decoy_dir + '/' + decoys[i])
                        cen_std = SwitchResidueTypeSetMover('centroid')
                        cen_std.apply(pdb)
                        
                        env_score = ScoreFunction()
                        env_score.set_weight(core.scoring.cen_env_smooth, 1.0)
                        env_score(pdb)
                        energies = pdb.energies()
                        per_res_env_score = [energies.residue_total_energy(r)
                                                        for r in range(1, pdb.total_residue() + 1)]
                        pair_score = ScoreFunction()
                        pair_score.set_weight(core.scoring.cen_pair_smooth, 1.0)
                        pair_score(pdb)
                        energies = pdb.energies()
                        per_res_pair_score = [energies.residue_total_energy(r)
                                                        for r in range(1, pdb.total_residue() + 1)]

                        cbeta_score = ScoreFunction()
                        cbeta_score.set_weight(core.scoring.cbeta_smooth, 1.0)
                        cbeta_score(pdb)
                        energies = pdb.energies()
                        per_res_cbeta_score = [energies.residue_total_energy(r)
                                                        for r in range(1, pdb.total_residue() + 1)]
                        vdw_score = ScoreFunction()
                        vdw_score.set_weight(core.scoring.vdw, 1.0)
                        vdw_score(pdb)
                        energies = pdb.energies()
                        per_res_vdw_score = [energies.residue_total_energy(r)
                                                        for r in range(1, pdb.total_residue() + 1)]
                        rg_score = ScoreFunction()
                        rg_score.set_weight(core.scoring.rg, 1.0)
                        rg_score(pdb)
                        energies = pdb.energies()
                        per_res_rg_score = [energies.residue_total_energy(r)
                                                        for r in range(1, pdb.total_residue() + 1)]
                        cenpack_score = ScoreFunction()
                        cenpack_score.set_weight(core.scoring.cenpack_smooth, 1.0)
                        cenpack_score(pdb)
                        energies = pdb.energies()
                        per_res_cenpack_score = [energies.residue_total_energy(r)
                                                        for r in range(1, pdb.total_residue() + 1)]
                        co_score = ScoreFunction()
                        co_score.set_weight(core.scoring.co, 1.0)
                        co_score(pdb)
                        energies = pdb.energies()
                        per_res_co_score = [energies.residue_total_energy(r)
                                                        for r in range(1, pdb.total_residue() + 1)]
                        hs_score = ScoreFunction()
                        hs_score.set_weight(core.scoring.hs_pair, 1.0)
                        hs_score(pdb)
                        energies = pdb.energies()
                        per_res_hs_score = [energies.residue_total_energy(r)
                                                        for r in range(1, pdb.total_residue() + 1)]
                        ss_score = ScoreFunction()
                        ss_score.set_weight(core.scoring.ss_pair, 1.0)
                        ss_score(pdb)
                        energies = pdb.energies()
                        per_res_ss_score = [energies.residue_total_energy(r)
                                                        for r in range(1, pdb.total_residue() + 1)]
                        
                        rsigma_score = ScoreFunction()
                        rsigma_score.set_weight(core.scoring.rsigma, 1.0)
                        rsigma_score(pdb)
                        energies = pdb.energies()
                        per_res_rsigma_score = [energies.residue_total_energy(r)
                                                        for r in range(1, pdb.total_residue() + 1)]
                        
                        sheet_score = ScoreFunction()
                        sheet_score.set_weight(core.scoring.sheet, 1.0)
                        sheet_score(pdb)
                        energies = pdb.energies()
                        per_res_sheet_score = [energies.residue_total_energy(r)
                                                        for r in range(1, pdb.total_residue() + 1)]
                        
                        cen_hb_score = ScoreFunction()
                        cen_hb_score.set_weight(core.scoring.cen_hb, 1.0)
                        cen_hb_score(pdb)
                        energies = pdb.energies()
                        per_res_cen_hb_score = [energies.residue_total_energy(r)
                                                        for r in range(1, pdb.total_residue() + 1)]

                        #write output#
                        outFile = open(output_path + '/' + decoys[i].split('.')[0] +".rosetta", "w")
                        for k in range(len(residueList)):
                                outFile.write(str(residueList[k]) + " " +str(per_res_env_score[k]) + " " + str(per_res_pair_score[k]) + " " + str(per_res_cbeta_score[k]) + " " +
                                              str(per_res_vdw_score[k]) + " " + str(per_res_rg_score[k]) + " " + str(per_res_cenpack_score[k]) + " " + str(per_res_co_score[k]) + " " +
                                              str(per_res_hs_score[k]) + " " + str(per_res_ss_score[k]) + " " + str(per_res_rsigma_score[k]) + " " + str(per_res_sheet_score[k]) + " " +
                                              str(per_res_cen_hb_score[k]) + "\n")  

                except Exception as e:
                        print("Error occured while generating rosetta energy\n" + str(e))

