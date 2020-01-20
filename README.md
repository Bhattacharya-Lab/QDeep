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
<br/>
<p>
  <p style="margin: 0px; font-stretch: normal; font-size: 11px; line-height: normal; font-family: Menlo; background-color: rgb(255, 255, 255);"><span style="font-variant-ligatures: no-common-ligatures;">**********************************************************************</span></p>
<p style="margin: 0px; font-stretch: normal; font-size: 11px; line-height: normal; font-family: Menlo; background-color: rgb(255, 255, 255);"><span style="font-variant-ligatures: no-common-ligatures;">* &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;QDeep &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; *</span></p>
<p style="margin: 0px; font-stretch: normal; font-size: 11px; line-height: normal; font-family: Menlo; background-color: rgb(255, 255, 255);"><span style="font-variant-ligatures: no-common-ligatures;">* &nbsp; &nbsp; &nbsp; &nbsp;Protein sinlge-model quality assessment using ResNet &nbsp; &nbsp; &nbsp; &nbsp;*</span></p>
<p style="margin: 0px; font-stretch: normal; font-size: 11px; line-height: normal; font-family: Menlo; background-color: rgb(255, 255, 255);"><span style="font-variant-ligatures: no-common-ligatures;">* &nbsp; &nbsp; &nbsp; For comments, please email to bhattacharyad@auburn.edu &nbsp; &nbsp; &nbsp; *</span></p>
<p style="margin: 0px; font-stretch: normal; font-size: 11px; line-height: normal; font-family: Menlo; background-color: rgb(255, 255, 255);"><span style="font-variant-ligatures: no-common-ligatures;">**********************************************************************</span></p>
<p style="margin: 0px; font-stretch: normal; font-size: 11px; line-height: normal; font-family: Menlo; background-color: rgb(255, 255, 255);"><span style="font-variant-ligatures: no-common-ligatures;">usage: QDeep.py [-h] [--tar TARGET_NAME] [--fas FASTA_FILE] [--dec DECOY_DIR]</span></p>
<p style="margin: 0px; font-stretch: normal; font-size: 11px; line-height: normal; font-family: Menlo; background-color: rgb(255, 255, 255);"><span style="font-variant-ligatures: no-common-ligatures;">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [--aln ALN_FILE] [--dist DISTANCE_FILE] [--pssm PSSM_FILE]</span></p>
<p style="margin: 0px; font-stretch: normal; font-size: 11px; line-height: normal; font-family: Menlo; background-color: rgb(255, 255, 255);"><span style="font-variant-ligatures: no-common-ligatures;">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [--spd SPD33_FILE] [--msa YES] [--gpu DEVICE_ID]</span></p>
<p style="margin: 0px; font-stretch: normal; font-size: 11px; line-height: normal; font-family: Menlo; background-color: rgb(255, 255, 255);"><span style="font-variant-ligatures: no-common-ligatures;">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; [--out OUTPUT_PATH]</span></p>
<p style="margin: 0px; font-stretch: normal; font-size: 11px; line-height: normal; font-family: Menlo; background-color: rgb(255, 255, 255); min-height: 13px;">
  <br>
</p>
<p style="margin: 0px; font-stretch: normal; font-size: 11px; line-height: normal; font-family: Menlo; background-color: rgb(255, 255, 255);"><span style="font-variant-ligatures: no-common-ligatures;">optional arguments:</span></p>
<p style="margin: 0px; font-stretch: normal; font-size: 11px; line-height: normal; font-family: Menlo; background-color: rgb(255, 255, 255);"><span style="font-variant-ligatures: no-common-ligatures;">&nbsp; -h, --help &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;show this help message and exit</span></p>
<p style="margin: 0px; font-stretch: normal; font-size: 11px; line-height: normal; font-family: Menlo; background-color: rgb(255, 255, 255);"><span style="font-variant-ligatures: no-common-ligatures;">&nbsp; --tar TARGET_NAME &nbsp; &nbsp; Target name</span></p>
<p style="margin: 0px; font-stretch: normal; font-size: 11px; line-height: normal; font-family: Menlo; background-color: rgb(255, 255, 255);"><span style="font-variant-ligatures: no-common-ligatures;">&nbsp; --fas FASTA_FILE &nbsp; &nbsp; &nbsp;Fasta file</span></p>
<p style="margin: 0px; font-stretch: normal; font-size: 11px; line-height: normal; font-family: Menlo; background-color: rgb(255, 255, 255);"><span style="font-variant-ligatures: no-common-ligatures;">&nbsp; --dec DECOY_DIR &nbsp; &nbsp; &nbsp; Decoy directory</span></p>
<p style="margin: 0px; font-stretch: normal; font-size: 11px; line-height: normal; font-family: Menlo; background-color: rgb(255, 255, 255);"><span style="font-variant-ligatures: no-common-ligatures;">&nbsp; --aln ALN_FILE &nbsp; &nbsp; &nbsp; &nbsp;Multiple Sequence Alignment</span></p>
<p style="margin: 0px; font-stretch: normal; font-size: 11px; line-height: normal; font-family: Menlo; background-color: rgb(255, 255, 255);"><span style="font-variant-ligatures: no-common-ligatures;">&nbsp; --dist DISTANCE_FILE &nbsp;DMPfold predicted distance</span></p>
<p style="margin: 0px; font-stretch: normal; font-size: 11px; line-height: normal; font-family: Menlo; background-color: rgb(255, 255, 255);"><span style="font-variant-ligatures: no-common-ligatures;">&nbsp; --pssm PSSM_FILE &nbsp; &nbsp; &nbsp;PSSM file</span></p>
<p style="margin: 0px; font-stretch: normal; font-size: 11px; line-height: normal; font-family: Menlo; background-color: rgb(255, 255, 255);"><span style="font-variant-ligatures: no-common-ligatures;">&nbsp; --spd SPD33_FILE &nbsp; &nbsp; &nbsp;SPIDER3 output (.spd3)</span></p>
<p style="margin: 0px; font-stretch: normal; font-size: 11px; line-height: normal; font-family: Menlo; background-color: rgb(255, 255, 255);"><span style="font-variant-ligatures: no-common-ligatures;">&nbsp; --msa YES &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; yes|no Whether to use deep MSA (default: no)</span></p>
<p style="margin: 0px; font-stretch: normal; font-size: 11px; line-height: normal; font-family: Menlo; background-color: rgb(255, 255, 255);"><span style="font-variant-ligatures: no-common-ligatures;">&nbsp; --gpu DEVICE_ID &nbsp; &nbsp; &nbsp; device id (0/1/2/3/4/..) Whether to run on GPU</span></p>
<p style="margin: 0px; font-stretch: normal; font-size: 11px; line-height: normal; font-family: Menlo; background-color: rgb(255, 255, 255);"><span style="font-variant-ligatures: no-common-ligatures;">&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; (default: CPU)</span></p>
<p style="margin: 0px; font-stretch: normal; font-size: 11px; line-height: normal; font-family: Menlo; background-color: rgb(255, 255, 255);"><span style="font-variant-ligatures: no-common-ligatures;">&nbsp; --out OUTPUT_PATH &nbsp; &nbsp; output directory name</span></p>
  </p>



<!-- LICENSE -->
## License

Distributed under the Apache 2.0 License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact




<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

