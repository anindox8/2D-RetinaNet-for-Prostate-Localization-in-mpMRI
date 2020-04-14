from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import pandas as pd
import SimpleITK as sitk
import os

'''
Binary PCa Classification in mpMRI
Script:         Redundancy Check (Max ValAUC-Epoch Check)
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         10/03/2020

'''

weights_path   = './models/delta/weights/68_14032020/'   
epoch_list     = []
val_loss_list  = []
val_auc_list   = []

# For Each Weights Checkpoint
for f in os.listdir(weights_path):
    if 'Epoch' in f:
        epoch         = str(f).split('_ValLoss')[0].split('Epoch-')[1]
        val_loss      = str(f).split('_ValAUC')[0].split('ValLoss-')[1]
        val_auc       = str(f).split('_ValAUC-')[1]

        # Populate Lists
        epoch_list.append(epoch)  
        val_loss_list.append(val_loss)
        val_auc_list.append(val_auc) 
  
# Export CSV Data
CSVdata      =  pd.DataFrame(list(zip(epoch_list,
                                      val_loss_list,
                                      val_auc_list)),
columns      =  ['epoch', 'val_loss', 'val_auc'])
CSVdata_name =  weights_path+'ValidationMetrics.csv'
CSVdata.to_csv(CSVdata_name, encoding='utf-8', index=False)

