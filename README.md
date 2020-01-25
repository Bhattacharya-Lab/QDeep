<!-- TITLE -->
<p align="center">
<h1 align = "center">QDeep</h2>
  </a>

  <h2 align="center">QDeep: distance-based protein model quality estimation by residue-level ensemble error classifications using stacked deep residual neural networks</h2>

<!-- TABLE OF CONTENTS -->
## Table of Contents
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [License](#license)
* [Acknowledgements](#acknowledgements)

## Getting Started

QDeep can be downloaded by typing
```
$ git clone https://github.com/Bhattacharya-Lab/QDeep.git
```

### Prerequisites

1. Linux system: QDeep is tested on x86_64 CentOs Linux 7 system. Currently, QDeep is not supported on Windows or Mac
2. python 3.6 or newer <br/>
3. tensorflow 1.13.1 or newer <br/>
4. keras 2.3.1 or newer <br/>
5. numpy 1.14.5 or newer <br/>
6. Pyrosetta Python-3.6.Release: pyrosetta-2019.45+release or newer (http://www.pyrosetta.org/dow) <br/>

### Installation
1. If you don't have python version 3.6, you can download from <a href="https://www.python.org/downloads/release/python-360/">https://www.python.org/downloads/release/python-360/</a> and install.
2. If you don't have "tensorflow" package, install it by typing ```$ pip install tensorflow```
3. If you don't have "keras" package, install it by typing ```$ pip install keras```
4. If you don't have "numpy" package, install it by typing ```$ pip install numpy```
5. If you don't have Pyrosetta installed, please download from <a href="http://www.pyrosetta.org/dow">http://www.pyrosetta.org/dow</a> and install. If it requires license, you can obtain the license from <a href="https://els.comotion.uw.edu/licenses/88">https://els.comotion.uw.edu/licenses/88</a>. You will receive an email with Username and Password. After you download and unzip, please use following commands to install:
```
$ cd setup
$ python setup.py install
```
6. Go to the directory where you download QDeep and configure by typing:
```
$ cd QDeep
$ python configure.py
```

<!--- USAGE---->
## Usage
To run QDeep, type
```
$ python QDeep.py
```
You will see the following output
```
***************************************************************************
*                               QDeep                                     *
*   Distance-based protein model quality estimation using deep ResNets    *
*          For comments, please email to bhattacharyad@auburn.edu         *
***************************************************************************
usage: QDeep.py [-h] [--tar TARGET_NAME] [--seq SEQ_FILE] [--dcy DECOY_DIR]
                [--aln ALN_FILE] [--dist DISTANCE_FILE] [--pssm PSSM_FILE]
                [--spd3 SPD33_FILE] [--msa YES] [--gpu DEVICE_ID]
                [--out OUTPUT_PATH]

Arguments:
  -h, --help            show this help message and exit
  --tar TARGET_NAME     Target name
  --seq SEQ_FILE        Sequence file
  --dcy DECOY_DIR       Decoy directory
  --aln ALN_FILE        Multiple sequence alignment
  --dist DISTANCE_FILE  DMPfold predicted distance
  --pssm PSSM_FILE      PSSM file
  --spd3 SPD33_FILE     SPIDER3 output (.spd3)
  --msa YES             yes|no Whether to use deep MSA (default: no)
  --gpu DEVICE_ID       device id (0/1/2/3/4/..) Whether to run on GPU
                        (default: CPU)
  --out OUTPUT_PATH     output directory name
```
<b>Example commands to run QDeep</b><br/><br/>
QDeep can be run with both standard and deep MSA.</br>
* To run QDeep with standard MSA, type
```
$ cd QDeep
$ python QDeep.py --tar T0865 --fas example/T0865.fasta --dec example/T0865 --aln example/T0865.aln --dist example/rawdistpred.current --pssm example/T0865.pssm --spd example/T0865.spd33 --out T0865
```
   Please check the <a href="https://github.com/Bhattacharya-Lab/QDeep/blob/master/run.log">log</a> to match with your output for the above command.
* To run QDeep with deep MSA, type
```
$ cd QDeep
$ python QDeep.py --tar T0865 --fas example/T0865.fasta --dec example/T0865 --aln example/T0865.aln --dist example/rawdistpred.current --pssm example/T0865.pssm --spd example/T0865.spd33 --msa yes --out T0865
```
* For running QDeep, GPU is not required. However GPU may faster the prediction. To run QDeep with GPU, type
```
$ cd QDeep
$ python QDeep.py --tar T0865 --fas example/T0865.fasta --dec example/T0865 --aln example/T0865.aln --dist example/rawdistpred.current --pssm example/T0865.pssm --spd example/T0865.spd33 --msa yes --gpu 0 --out T0865
```

A detailed explanation for each of the options are provided below<br/>
* --tar Target name: This should be the name of target without having any extension.<br/>
* --fas Fasta file: This should contain the sequence with and without the header. The sequence may also expands to multiple lines in the fasta file<br/>
* --dec Decoy directory: This requires a directory containing all the pdb models with .pdb extension.<br/>
* --aln Multiple Sequence Alignment file: The alignment file should be generated using HHblits with a query sequence coverage of 10% and pairwise sequence identity of 90% against uniclust30_2018_08 by three iterations with an E-value inclusion threshold of 10^-3. You can download HHblits from https://github.com/soedinglab/hh-suite.<br/>
* --dist DMPfold predicted distance: To predict distance using DMPfold, you can download DMPfold from https://github.com/psipred/DMPfold<br/>
* --pssm PSSM file: You can generate the sequence profile by searching the NR database using PSI-BLAST. You can download the PSI-BLAST from ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/<br/>
* --spd SPIDER3 output: The secondary structure and the solvent accessibility should be predicted using SPIDER3. SPIDER3 can be downloaded from https://sparks-lab.org/downloads/
* --msa yes|no: This is optional. You should use this flag if you want to use DeepMSA generated MSA. You can download DeepMSA from https://zhanglab.ccmb.med.umich.edu/DeepMSA/. When you use DeepMSA generated MSA, please make sure to, 
  * generate sequene profile(PSSM) using deep MSA
  * run SPIDER3 using deep MSA to predict secondary strcture and solvent accessibility 
  * predict distance using DMPfold using deep MSA
  
* --gpu device_id: If you want to use GPU for prediction, please use this flag and specify the device ID.
* --out output location: Please select a location for the output to be stored. It is recommended that you specify a directory name for the output.

## CITE

If you find QDeep useful, please cite our paper
