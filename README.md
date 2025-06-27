# Graph Neural Network for pre-selection of solvents involving neutral molecules and ionic liquids

![CI](https://github.com/A-JMinor/GNN_IL_solvent_screening/actions/workflows/test.yml/badge.svg)


## Overview

This repository contains the code used for the solvent pre-selection analysis presented in the paper [Solvent Screening and Conceptual Process Design for the Chemical Recycling of Nylon 6 to Caprolactam Using Ionic Liquids](). This work makes use of an extended Gibbs-Helmholtz Graph Neural Network that predicts infinite dilution activity coefficients of systems that include neutral and ionic molecules.

## Repository Structure

```
├── data/                   # Solvents considered during screening and pure-component properties from KDB
├── models/                 # Implementation of GNN-based models
├── src/                    # Source code for solvent selection + utilities
├── requirements.txt        # Required Python dependencies
├── README.md               # This document
├── LICENSE                 # License information
└── run.bat                 # Batch script to execute analysis
```

## Installation

### Prerequisites

Ensure you have Python installed (we tested with 3.10). It is advised to create a virtual environment before installing dependencies.

```
python -m venv gnn_solv_screen
gnn_solv_screen\Scripts\activate
```

Then, clone this repository and enter the working folder:

```
git clone https://github.com/A-JMinor/GNN_IL_solvent_screening.git
cd GNN_IL_solvent_screening
```

### Install Dependencies

```
pip install -r requirements.txt
```

or simply do

```
conda create --name venv --file requirements.txt
```

## Usage

The `run.bat` script automates all steps of the solvent pre-selection

To execute simply run:

```
./run.bat
```

## Citation

If you use this repository in your research, please cite our paper:

```

```

## Related works

You might be interested in checking other related works:
- [Solvent pre-selection for extractive distillation using GNNs](https://doi.org/10.1016/B978-0-443-15274-0.50324-3)
- A systematic comparison between solvent rankings of GNN vs. UNIFAC-IL
- [Gibbs-Helmholtz Graph Neural Network](https://doi.org/10.1039/D2DD00142J)
- [Gibbs-Helmholtz Graph Neural Network extended to polymer solutions](https://doi.org/10.1021/acs.jpca.3c05892)



## License

This project is licensed under the [MIT License](https://github.com/A-JMinor/GNN_IL_solvent_screening/blob/main/LICENSE) - see the LICENSE file for details.

## Contact

For questions or collaborations, please feel free to reach out.