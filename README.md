# Predicting Intelligence from Brain Gray Matter Volume
Hilger, Winter et al. (2019)

#### Project Phase
submitted to Brain Structure and Function
#### Description
The following repository provides a detailed description of all Python scripts used in the paper "Prediction Intelligence 
from Brain Gray Matter Volume". The scripts within this repository can be used to replicate the published results.

## 1 Data
In this study, we used T1-weighted MR images aquired by the Nathan S. Kline Institute for Psychiatric Research (NKI) and 
released as part of the 1000 functional connectomes project (1).

Data can be downloaded from: 
http://www.nitrc.org/ir/app/template/XDATScreen_report_xnat_projectData.vm/search_element/xnat:projectData/search_field/xnat:projectData.ID/search_value/nki_rockland
For more detailed information see: http://fcon_1000.projects.nitrc.org/indi/pro/nki.html


## 2 Preprocessing
We generated individual maps of regional gray matter volume with the CAT12 toolbox (Computational Anatomy Toolbox 
version 10.73; http://www.neuro.uni-jena.de/cat/) for SPM12 (Statistic Parametric Mapping software, Welcome Department 
of Imaging Neuroscience, London, UK).

As we were interested in relative gray matter volume, i.e., individual differences in gray matter volume that cannot be 
attributed to individual differences in brain size, the m-modulated gray matter probability maps were then corrected 
for total intracranial volume (TIV) by global rescaling. Rescaling is recommended (Gaser & Kurth, 2018) when TIV 
significantly correlates with the variable of interest, i.e., the targets of the prediction models. 
To do this, we calculated each subject’s TIV and rescaled the gray matter value of each voxel by (1) dividing it by the 
subject’s individual TIV value and then (2) multiplying the result with the mean TIV value of the whole group. 

After correcting for individual TIV values, we organised the data in two folders, one containing the corrected and one 
the uncorrected gray matter maps.

## 3 Software and Hardware Requirements
### Software
For all analyses, we used the publicly available machine learning Python toolbox PHOTON 
([PHOTON](https://github.com/photon-team/photon)). To install the PHOTON version used here, do
```
pip install git+https://github.com/photon-team/photon@predicting-intelligence-from-brain-gray-matter
```
### Hardware
Since running the permutation test for all analyses is quite resource intense, we used a High-Performance-Cluster (HPC) 
with SLURM job scheduling. The scripts for the permutation tests provided here open a SLURM task array with 1000 tasks, 
each representing a permutation run. If you want to run a permutation test without a cluster, you can do so by using the 
PermutationTest() class within PHOTON. However, this will probably take weeks to finish so using a cluster is highly 
recommended.

## 4 Analysis
### Retrieving the data
The module data.py provides a helper class IQData() that encapsulates the data loading procedure to simplify the use of 
multiple scripts for every part of the analyses. When IQData() is instantiated, the subjects' information is automatically 
loaded and saved within the object itself. By using load_whole_brain() and load_single_networks(), the nifti images are 
loaded using PHOTON Neuro. A BrainAtlas PipelineElement then takes care of loading the nifti images and applying an 
atlas to retrieve either the whole brain or networks data. To separate between the tiv-corrected and uncorrected data,
one can simply pass a boolean to tiv_rescaled when instantiating the IQData object.

### Constructing a PHOTON hyperpipe

### Global Analysis (Whole Brain) 

### Local Analysis (Networks)

## References