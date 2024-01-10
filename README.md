# ProNEP
![Alt text](<Model frame .png>)
### ProNEP is an computational method designed for the precise identification of Correspondence between NLRs (Nucleotide-binding leucine-rich repeat receptors) and effectors.  Leveraging the power of transfer learning and a sophisticated bilinear attention network, ProNEP is implemented within the PyTorch framework, utilizing protein sequences as input.
## Follow these steps to get started with ProNEP：
ProNEP has been transformed into an easy-to-use webserver and is now freely accessible at [NERRD](http://nerrd.cn/#/prediction).
### 1. Request Environment
conda install h5py

conda install numpy

conda install pandas

conda install sys

conda install tqdm

conda install scikit-learn

conda install prettytable

conda install argparse

Pytorch = 2.1.0
### 2. Data
The Datasets folder contains all the data used in the experiment. The example folder is the experimental data for ProNEP testing.
### 3. run
To run the ProNEP script, please download, unzip [saved-models](http://nerrd.cn/#/help) and move the contains to ./
#### 1. Predict the result of folder example
python pre.py --file1 ./example/CSA1.fasta --file2 ./example/HopB1.fasta


The example folder is based on the paper by Schulze in 2022<sup><a href="#ref1">1</a></sup>.

#### 2. Train
python run.py --cfg "configs/ProNEP.yaml" --data nlr

If you intend to train on your own dataset, make sure to remember to modify the file names in the run.py file.
#### 3. Predict
python pre.py --file1 yourNLRdata --file2 youreffectordata

#### Reference
1. <p name = "ref1">Schulze, S., Yu, L., Hua, C., Zhang, L., Kolb, D., Weber, H., Ehinger, A., Saile, S.C., Stahl, M., Franz-Wachtel, M., et al. (2022). The Arabidopsis TIR-NBS-LRR protein CSA1 guards BAK1-BIR3 homeostasis and mediates convergence of pattern- and effector-induced immune responses. Cell host microbe 30, 1717-1731.</p>





