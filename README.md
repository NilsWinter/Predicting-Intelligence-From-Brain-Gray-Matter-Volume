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
[NITRC NKI DATA](http://www.nitrc.org/ir/app/template/XDATScreen_report_xnat_projectData.vm/search_element/xnat:projectData/search_field/xnat:projectData.ID/search_value/nki_rockland)

For more detailed information see: 
[1000 Functional Connectome Project](http://fcon_1000.projects.nitrc.org/indi/pro/nki.html)


## 2 Preprocessing
We generated individual maps of regional gray matter volume with the CAT12 toolbox (Computational Anatomy Toolbox 
version 10.73; http://www.neuro.uni-jena.de/cat/) for SPM12 (Statistic Parametric Mapping software, Welcome Department 
of Imaging Neuroscience, London, UK).

As we were interested in relative gray matter volume, i.e., individual differences in gray matter volume that cannot be 
attributed to individual differences in brain size, the m-modulated gray matter probability maps were then corrected 
for total intracranial volume (TIV) by global rescaling. Rescaling is recommended (Gaser & Kurth, 2018) when TIV 
significantly correlates with the variable of interest, i.e., the targets of the prediction models. 
To do this, we calculated each subject’s TIV and rescaled the gray matter value of each voxel by first dividing it by the 
subject’s individual TIV value and then multiplying the result with the mean TIV value of the whole group. 

After correcting for individual TIV values, we organised the data in two folders, one containing the corrected and one 
the uncorrected gray matter maps.

## 3 Software and Hardware Requirements
### Software
For all analyses, we used the publicly available machine learning Python toolbox 
[PHOTON](https://github.com/photon-team/photon). To install the PHOTON version used here, do
```
pip install git+https://github.com/photon-team/photon@predicting-intelligence-from-brain-gray-matter
```
### Hardware
Since running the permutation test for all analyses is quite resource intense, we used a High-Performance-Cluster (HPC) 
with SLURM job scheduling. The scripts for the permutation tests provided here open a SLURM task array with 1000 tasks, 
each representing a permutation run. If you want to run a permutation test without a cluster, you can do so by using the 
PermutationTest() class within PHOTON. However, this will probably take weeks to finish so using a cluster is highly 
recommended.

## 4 Analysis Description
### Retrieving the data
The module data.py provides a helper class IQData() that encapsulates the data loading procedure to simplify the use of 
multiple scripts for every part of the analyses. When IQData() is instantiated, the subjects' information is automatically 
loaded and saved within the object itself. By using load_whole_brain() and load_single_networks(), the nifti images are 
loaded using PHOTON Neuro. A BrainAtlas PipelineElement then takes care of loading the nifti images and applying an 
atlas to retrieve either the whole brain or networks data. To separate between the tiv-corrected and uncorrected data,
one can simply pass a boolean to tiv_rescaled when instantiating the IQData object.

### Constructing a PHOTON hyperpipe
For a more detailed description of PHOTON, see www.photon-ai-com

In short, PHOTON creates a so-called hyperpipe object that optimizes hyperparameters of a machine learning pipeline in 
a nested cross-validated scheme using different optimization strategies.

Below you can see the hyperpipe definition that is used in all analyses. It includes a stratified K-fold regression 
cross-validation, a confounder removal step to regress out sex, handedness and age, a PCA that performs a full variance 
decompositon to reduce the dimensionality of the data and a Linear Support Vector Regression as final estimator. In a 
PHOTON hyperpipe, it is possible to define hyperparameters of a specific pipeline element by providing them as Python 
dictionary.
```python
# cv
outer_cv = StratifiedKFoldRegression(n_splits=10, shuffle=True, random_state=3)
inner_cv = StratifiedKFoldRegression(n_splits=3, shuffle=True, random_state=4)

# define output
output = OutputSettings(save_predictions='best', save_feature_importances='None', project_folder=project_folder)

# define hyperpipe
pipe = Hyperpipe(name=name,
                 optimizer='sk_opt',
                 optimizer_params={'num_iterations': 50, 'base_estimator': 'GP'},
                 metrics=['mean_squared_error', 'mean_absolute_error', 'variance_explained'],
                 best_config_metric='mean_squared_error',
                 outer_cv=outer_cv,
                 inner_cv=inner_cv,
                 eval_final_performance=True,
                 verbosity=1,
                 output_settings=output)

# add elements to hyperpipe
# add confounder removal first
pipe += PipelineElement('ConfounderRemoval', {}, standardize_covariates=True, cache_dir=cache_dir,
                        test_disabled=False)
pipe += PipelineElement('PhotonPCA', {}, n_components=None,  test_disabled=False,
                        logs=cache_dir)

pipe += PipelineElement('LinearSVR', {
    'C': FloatRange(0.000001, 1, range_type='linspace'),
    'epsilon': FloatRange(0.01, 3, range_type='linspace')})
```
### Global Analysis (Whole Brain) 

### Local Analysis (Networks)

## 5 How to run the scripts

## References