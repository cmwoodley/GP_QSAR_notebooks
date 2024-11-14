## Example notebooks for training models using GP_qsar and using the in-build active learning methods

### Installation instructions

These notebooks depend on REINVENT4 and GP_QSAR. These notebooks and environment have only been tested in WSL2 in Windows11

1. Create Conda environment
```
conda create -n gp_qsar python==3.10.15
conda activate gp_qsar
```
2. Clone and install REINVENT4
Instructions provided at the [REINVENT4 repository](https://github.com/MolecularAI/REINVENT4/)

3. Clone and install GP_QSAR
```
git clone https://github.com/cmwoodley/GP_qsar.git
cd GP_qsar
pip install .
```

4. Install further dependencies
```
conda install ipykernel
```
