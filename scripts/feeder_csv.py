import os
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import KFold


'''
Prostate Detection in mpMRI
Script:         Generate Feeder CSV (K-Fold)
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         06/04/2020

'''


# Feeding Script Input Parameters
organ_modality      = 'prostate-mpMRI-128'
dataset_path        = './dataset/numpy/patch_128/'
class_names         = ['benign', 'malignant']
label_signs         = [ 0, 1 ]
partitions          = ['training-fold', 'validation-fold']
save_path           =  './models/laufey/feed/'
folds               =   5


# Generating Feed Directories for Class Folds
for j in range(len(class_names)):
    image_list       = os.listdir(dataset_path+'images/'+class_names[j]+'/')
    class_label      = np.full(len(image_list), label_signs[j])
    subject_id_list  = []
    image_path_list  = []
    label_path_list  = []
    # Populating Lists
    for i in image_list:
        # Extracting Fields
        subject_id   = i.split('.npy')[0]
        image_path   = dataset_path+'images/'+class_names[j]+'/'+i
        label_path   = dataset_path+'labels/'+i
        # Populating Lists
        subject_id_list.append(subject_id)
        image_path_list.append(image_path)
        label_path_list.append(label_path)
    
    # Synchronous Data Shuffle
    a,b,c,d  = shuffle(subject_id_list, class_label, image_path_list, label_path_list, random_state=8)
    # Metadata Setup
    shuffled_dataset  = pd.DataFrame(list(zip(a,b,c,d)),
    columns           = ['subject_id','class_label','image_path_list','label_path_list'])
    subject_id        = shuffled_dataset['subject_id']
    class_label       = shuffled_dataset['class_label']
    image_path        = shuffled_dataset['image_path_list']
    label_path        = shuffled_dataset['label_path_list']
    
    # Generating CSV
    kf     = KFold(folds)
    fold   = 0
    for train, val in kf.split(a):
        fold   +=1
        a_train = subject_id[train]
        b_train = class_label[train]
        c_train = image_path[train]
        d_train = label_path[train]
        a_val   = subject_id[val]
        b_val   = class_label[val]
        c_val   = image_path[val]
        d_val   = label_path[val]

        trainData                  = pd.DataFrame(list(zip(a_train,b_train,c_train,d_train)),
        columns                    = ['subject_id','class_label','image_path','label_path'])
        trainData_name             = save_path+'{}_{}_{}-{}'.format(organ_modality,class_names[j],partitions[0],fold)+'.csv'
        trainData.to_csv(trainData_name, encoding='utf-8', index=False)
    
        valData                    = pd.DataFrame(list(zip(a_val,b_val,c_val,d_val)),
        columns                    = ['subject_id','class_label','image_path','label_path'])
        valData_name               = save_path + '/{}_{}_{}-{}'.format(organ_modality,class_names[j],partitions[1],fold)+'.csv'
        valData.to_csv(valData_name, encoding='utf-8', index=False)
        print('Complete: ',trainData_name,' ; ',valData_name)


# Consolidating Feed Directories for Complete Folds
for P in range(len(partitions)):
    for F in range(0,folds):   
        class0_fold = pd.read_csv(save_path+'{}_{}_{}-{}'.format(organ_modality,class_names[0],partitions[P],F+1)+'.csv',dtype=object,keep_default_na=False,na_values=[])
        class1_fold = pd.read_csv(save_path+'{}_{}_{}-{}'.format(organ_modality,class_names[1],partitions[P],F+1)+'.csv',dtype=object,keep_default_na=False,na_values=[])

        partition_fold_subject_id  = []
        partition_fold_class_label = []
        partition_fold_image_path  = []
        partition_fold_label_path  = []
    
        # Class 0 (Benign)
        partition_fold_subject_id.extend(class0_fold['subject_id'].tolist()) 
        partition_fold_class_label.extend(class0_fold['class_label'].tolist())
        partition_fold_image_path.extend(class0_fold['image_path'].tolist())
        partition_fold_label_path.extend(class0_fold['label_path'].tolist())
        # Class 1 (Malignant)
        partition_fold_subject_id.extend(class1_fold['subject_id'].tolist()) 
        partition_fold_class_label.extend(class1_fold['class_label'].tolist())
        partition_fold_image_path.extend(class1_fold['image_path'].tolist())
        partition_fold_label_path.extend(class1_fold['label_path'].tolist())
    
        # Synchronous Data Shuffle and CSV Export
        a,b,c,d        =  shuffle(partition_fold_subject_id, partition_fold_class_label, 
                                  partition_fold_image_path, partition_fold_label_path, random_state=8)
        partitionData  =  pd.DataFrame(list(zip(a,b,c,d)),
        columns        =  ['subject_id','class_label','image_path','label_path'])
        partitionData.to_csv(save_path + '/{}_{}-{}'.format(organ_modality,partitions[P],F+1)+'.csv', encoding='utf-8', index=False)    
