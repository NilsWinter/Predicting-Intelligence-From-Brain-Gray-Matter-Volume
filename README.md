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

## 3 Software Requirements

## 4 ML PHOTON Base Analysis

## 5 Global Analysis (Whole Brain) 

## 6 Local Analysis (Networks)

## References