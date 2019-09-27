"""
===========================================================
Project: Predicting Intelligence from Brain Gray Matter
===========================================================
Description
-----------
This script specifies all relevant data for this project and provides the IQData class that is able to return the data.

Version
-------
Created:        28-01-2019
Last updated:   27-09-2019


Author
------
Nils R. Winter
nils.r.winter@gmail.com
Translationale Psychiatrie
Universitaetsklinikum Muenster
"""
import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from photonai.base.PhotonBase import PipelineElement
from photonai.neuro.AtlasStacker import AtlasInfo


class IQData:
    def __init__(self, data_folder: str = '../../', tiv_rescaled=True):
        self.folder = data_folder
        self.tiv_rescaled = tiv_rescaled
        self.whole_brain = None
        self.networks = None
        self.networks_as_matrix = None
        self.filter_indices = None

        # Load IQ scores
        df = pd.read_csv(self.folder + 'data/rockland_fsiq_scores.csv')
        df.rename(index=str, columns={'Ids': 'ID'}, inplace=True)

        # Load covariates
        covariates = pd.read_csv(self.folder + 'data/rockland_covariates.csv', sep=';', na_values=' ', decimal=',')[:308]
        df = df.merge(covariates, left_on='ID', right_on='ID', how='outer')

        # Assign all covariates to class variables to use later
        self.id = df['ID'].tolist()
        self.fsiq = np.asarray(df['FSIQ'].tolist())
        self.age = df['age'].values
        self.gender = df['sex'].values
        self.handedness = df['hand1'].fillna(np.mean(df['hand1'])).values
        self.tiv = df['TiV'].values
        self.n = len(self.id)

        # Get paths to rescaled and unscaled VBM data
        self.vbm_paths = list()
        for subject in self.id:
            if self.tiv_rescaled:
                self.vbm_paths.append(
                    self.folder + 'data/vbm_tiv_corrected/' + subject + '/mri/mwp1anatomical_scan_TIV_rescaled_finish.nii.gz')
            else:
                self.vbm_paths.append(
                    self.folder + 'data/vbm/' + subject + '/mri/mwp1anatomical_scan.nii')

    def load_whole_brain(self, use_cached=False):
        if not use_cached:
            atlas_info = AtlasInfo(atlas_name='mni_icbm152_t1_tal_nlin_sym_09a_mask', mask_threshold=.5,
                                   roi_names='all', extraction_mode='vec')

            atlas = PipelineElement('BrainAtlas', {}, atlas_info_object=atlas_info)
            self.whole_brain, _, _ = atlas.transform(self.vbm_paths)

            if not os.path.exists(self.folder + 'data/cache/'):
                os.mkdir(self.folder + 'data/cache/')
            if self.tiv_rescaled:
                np.save(self.folder + 'data/cache/whole_brain_residuals.npy', self.whole_brain)
            else:
                np.save(self.folder + 'data/cache/whole_brain_residuals_no_tiv_rescaling.npy', self.whole_brain)
        else:
            if self.tiv_rescaled:
                self.whole_brain = np.load(self.folder + 'data/cache/whole_brain_residuals.npy')
            else:
                self.whole_brain = np.load(self.folder + 'data/cache/whole_brain_residuals_no_tiv_rescaling.npy')


    def load_single_networks(self, use_cached=False):
        if not use_cached:
            # load yeo rois
            atlas_info = AtlasInfo(atlas_name='Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask',
                                   roi_names='all',
                                   extraction_mode='vec')
            atlas = PipelineElement('BrainAtlas', {}, atlas_info_object=atlas_info)
            rois, _, _ = atlas.transform(self.vbm_paths)

            # add subcortical
            atlas_info = AtlasInfo(atlas_name='VBM_AAL_mitWFU_higher_res_subcort_TH-HC_A_Cau_Pu',
                                   roi_names='all',
                                   extraction_mode='vec')
            atlas = PipelineElement('BrainAtlas', {}, atlas_info_object=atlas_info)
            rois.append(atlas.transform(self.vbm_paths)[0])

            # add cerebellum
            atlas_info = AtlasInfo(atlas_name='VBM_Cerebellum_from_AAL',
                                   roi_names='all',
                                   extraction_mode='vec')
            atlas = PipelineElement('BrainAtlas', {}, atlas_info_object=atlas_info)
            rois.append(atlas.transform(self.vbm_paths)[0])

            self.networks = rois

            X = np.empty((self.n, 0))
            filter_indices = OrderedDict()
            for i, network in enumerate(self.networks):
                filter_indices['network_' + str(i)] = np.arange(X.shape[1], X.shape[1] + network.shape[1])
                X = np.concatenate([X, network], axis=1)
                print('Network {} Number of Voxels {}'.format(i, network.shape[1]))

            self.networks_as_matrix = X
            self.filter_indices = filter_indices

            if not os.path.exists(self.folder + 'data/cache/'):
                os.mkdir(self.folder + 'data/cache/')
            for i, network in enumerate(self.networks):
                if self.tiv_rescaled:
                    np.save(self.folder + 'data/cache/network_' + str(i) + '_residuals.npy', network)
                else:
                    np.save(self.folder + 'data/cache/network_' + str(i) + '_residuals_no_tiv_rescaling.npy', network)
            np.save(self.folder + 'data/cache/filter_indices.npy', self.filter_indices)
            if self.tiv_rescaled:
                np.save(self.folder + 'data/cache/networks_residuals.npy', self.networks_as_matrix)
            else:
                np.save(self.folder + 'data/cache/networks_residuals_no_tiv_rescaling.npy', self.networks_as_matrix)
        else:
            self.networks = list()
            for i in range(9):
                if self.tiv_rescaled:
                    self.networks.append(np.load(self.folder + 'data/cache/network_' + str(i) + '_residuals.npy'))
                else:
                    self.networks.append(np.load(self.folder + 'data/cache/network_' + str(i) +
                                                 '_residuals_no_tiv_rescaling.npy'))
            if self.tiv_rescaled:
                self.networks_as_matrix = np.load(self.folder + 'data/cache/networks_residuals.npy')
            else:
                self.networks_as_matrix = np.load(self.folder + 'data/cache/networks_residuals_no_tiv_rescaling.npy')
            self.filter_indices = np.load(self.folder + 'data/cache/filter_indices.npy')
