# SMART-SC project
## Overview
Data analysis of the SMART-SC project.

* `analysis` contains the notebooks that cleans the data and make the plots
* `data` contains the raw and the processed data (in various formats)
* `reports` contains the generated figures, plots and other analysis

## Requirements
Conda (or even better mamba, the faster drop-in alternative)

## Install
1. Clone the repository
```
git clone https://github.com/andreagoo8/smartsc.git
cd smartsc
```

2. Create the smartsc environment with all the dependencies:
```
conda env create -f environment.lock.yml
conda deactivate && conda activate smartsc
```

3. Set up the folders:
```
mkdir -p data/{raw,processed/{excels,jsons-pp-2,jsons-pp-4,parquets,pickles/pps}}
mkdir -p reports/figures/{1-escribe/errors,4-eumaco-bpt,4-eumaco-spo2,4-eumaco-wt}
```

4. Initialize the project
```
pip install -e .
```
