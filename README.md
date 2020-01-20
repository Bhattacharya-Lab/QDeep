<!-- TITLE -->
<br />
<p align="center">
<h1 align = "center">QDeep</h2>
  </a>

  <h2 align="center">Distance-based protein model quality estimation by residue-level ensemble error classifications with stacked deep residual neural networks</h2>
  <p align="center">Md Hossain Shuvo (mzs0149@auburn.edu)<br/>
  Sutanu Bhattacharya (szb0134@auburn.edu)<br/>
  Debswapna Bhattacharya (bhattacharyad@auburn.edu)<br/>
  Last updated: 1/20/2020</p><br/>
</p>
<br />

<!-- TABLE OF CONTENTS -->
## Table of Contents
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Dependencies](#dependencies)
  * [Installation](#installation)
* [Usage](#usage)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)

## Getting Started

You can run QDeep via the <a href="http://watson.cse.eng.auburn.edu/QDeep">QDeep web server</a> to score a single protein model at a time. But if you need to evaluate a pool of models at a time, we strongly recommend that you download and run the tool locally on the Linux system using the following procedures. 

### Prerequisites

1. Any Linux system. Currently, QDeep is not supported on Windows or Mac
2. python==3.6 <br/>
3. tensorflow==1.13.1 <br/>
4. keras==2.3.1 <br/>
5. Pyrosetta Python-3.6.Release: pyrosetta-2019.45+release or newer (http://www.pyrosetta.org/dow) <br/>

### Dependencies

### Installation
1. If you don't have python version 3.6, you can download from <a href="https://www.python.org/downloads/release/python-360/">https://www.python.org/downloads/release/python-360/</a> and install.
2. If you don't have "tensorflow" package, install it by typing ```sh pip install tensorflow==1.13.1```
3. If you don't have "keras" package, install it by typing ```sh pip install keras==2.3.1```
4. If you don't have Pyrosetta installed, please download from <a href="http://www.pyrosetta.org/dow">http://www.pyrosetta.org/dow</a> and install. If it requires license, you can obtain the license from <a href="https://els.comotion.uw.edu/licenses/88">https://els.comotion.uw.edu/licenses/88</a>. You will receive an email with Username and Password.
5. Go to the directory where you download QDeep and configure by typing
```sh
$ python configure.py
```

<!--- USAGE---->
## Usage
To run QDeep, type
```sh
$ python QDeep.py
```
You will see the following output
```
**********************************************************************
*                            QDeep                                   *
*        Protein sinlge-model quality assessment using ResNet        *
*       For comments, please email to bhattacharyad@auburn.edu       *
**********************************************************************
usage: QDeep.py [-h] [--tar TARGET_NAME] [--fas FASTA_FILE] [--dec DECOY_DIR]
                [--aln ALN_FILE] [--dist DISTANCE_FILE] [--pssm PSSM_FILE]
                [--spd SPD33_FILE] [--msa YES] [--gpu DEVICE_ID]
                [--out OUTPUT_PATH]

optional arguments:
  -h, --help            show this help message and exit
  --tar TARGET_NAME     Target name
  --fas FASTA_FILE      Fasta file
  --dec DECOY_DIR       Decoy directory
  --aln ALN_FILE        Multiple Sequence Alignment
  --dist DISTANCE_FILE  DMPfold predicted distance
  --pssm PSSM_FILE      PSSM file
  --spd SPD33_FILE      SPIDER3 output (.spd3)
  --msa YES             yes|no Whether to use deep MSA (default: no)
  --gpu DEVICE_ID       device id (0/1/2/3/4/..) Whether to run on GPU
                        (default: CPU)
  --out OUTPUT_PATH     output directory name
```
<b>Example commands to run QDeep</b><br/><br/>
QDeep can be run with both shallow and deep MSA.</br>
* To run QDeep with shallow MSA, type
```
$ python QDeep.py --tar T0865 --fas example/T0865.fasta --dec example/T0865 --aln example/T0865.aln 
--dist example/rawdistpred.current --pssm example/T0865.pssm --spd example/T0865.spd33 --out T0865
```
* To run QDeep with deep MSA, type
```
$ python QDeep.py --tar T0865 --fas example/T0865.fasta --dec example/T0865 --aln example/T0865.aln 
--dist example/rawdistpred.current --pssm example/T0865.pssm --spd example/T0865.spd33 --msa yes --out T0865
```
* For running QDeep, GPU is not required. However GPU may faster the prediction. To run QDeep with GPU, type
```
$ python QDeep.py --tar T0865 --fas example/T0865.fasta --dec example/T0865 --aln example/T0865.aln 
--dist example/rawdistpred.current --pssm example/T0865.pssm --spd example/T0865.spd33 --msa yes --gpu 0 --out T0865
```

A detailed explanation for each of the options are provided below<br/>
* --tar Target name: This should be the name of target without having any extension.<br/>
* --fas Fasta file: This should contain the sequence with and without the header. The sequence may also expands to multiple lines in the fasta file<br/>
* --dec Decoy directory: This requires a directory containing all the pdb models with .pdb extension. Please note that, even if you score one pdb model, that should go in a directory to be passed.<br/>
* --aln Multiple Sequence Alignment file: The alignment file should be generated using HHblits with a query sequence coverage of 10% and pairwise sequence identity of 90% against uniclust30_2018_08 by three iterations with an E-value inclusion threshold of 10-3. You can download HHblits from https://github.com/soedinglab/hh-suite.<br/>
* --dist DMPfold predicted distance: To predict distance using DMPfold, you can download DMPfold from https://github.com/psipred/DMPfold<br/>
* --pssm PSSM file: You can generate the sequence profile by searching the NR database using PSI-BLAST. You can download the PSI-BLAST from ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/<br/>
* --spd SPIDER3 output: The secondary structure and the solvent accessibility should be predicted using SPIDER3. SPIDER3 can be downloaded from https://sparks-lab.org/downloads/
* --msa yes|no: This is optional. You should use this flag if you want to use DeepMSA generated MSA. When you use deep MSA, please make sure to, 
  * generate sequene profile(PSSM) using deep MSA
  * run SPIDER3 using deep MSA to predict secondary strcture and solvent accessibility 
  * predict distance using DMPfold using deep MSA
  
* --gpu device_id: If you want to use GPU for prediction, please use this flag and specify the device ID.
* --out output location: Please select a location for the output to be stored. It is recommended that you specify a directory name for the output.

<!-- LICENSE -->
## License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

