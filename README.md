# MRUS_registration
## Project description
This is a project on MR to US registration. The project uses two methods: Localnet and non-rigid registration
### Environment Requirements
```bash
    conda create -n sitk python==3.6
    conda activate sitk
```
- The necessary dependencies are listed in the requirements.txt file.
```bash
- pip install -r requirements.txt
```
- For the installSitk install,try the following command:
```bash
  pip install SimpleITK
```
- Also,you can refer to the https://github.com/SimpleITK/SimpleITK 
### Dataset preparation
- visit https://zenodo.org/records/8004388 to get the dataset.
### Running the Project
1. train localnet
```bash
   cd utils
   python train_localnet.py
```
2. For non_rigid registration
```bash
   cd utils
   python model_nonrigid.py
```

# Initial commit
# Initial commit
