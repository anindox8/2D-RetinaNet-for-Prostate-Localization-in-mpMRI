# 2D-RetinaNet-for-Prostate-Detection-in-mpMRI
 
**Problem Statement**: Localization of the prostate (healthy or with benign/malignant tumors) in multi-parametric MRI scans (T2W, DWI *with high b-value*, ADC).   

**Data**: 1950 prostate mpMRI volumes (*Healthy/Benign Cases*: 1234; *Malignant Cases*: 716); equivalent to 23400 2D slices. [1559/391: Train/Val Ratio] 

**Acknowledgments**: The following approach is based on the TensorFlow Estimator/Keras adaptation of [keras-retinanet](https://github.com/fizyr/keras-retinanet/) by Fizyr.

**Directories**  
  ● Preprocess Dataset to Spatially Resampled, Intensity Normalized, Overlapping (15%) Octant Patches in Optimized NumPy Format: `preprocess/prime/preprocess_deploy.py`  
  ● Generate Data-Directory Feeder List: `feed/prime/feed_metadata.py`  
  ● Train 2D RetinaNet Model: `train/prime/train_StFA.py`  
  ● Deploy Model (Validation): `deploy/prime/deployBinary.py`  
  


**Related Publication(s):**  
  ● A. Saha, F.I. Tushar, K. Faryna, V.D. Anniballe, R. Hou, M.A. Mazurowski, G.D. Rubin, J.Y. Lo (2020), "Weakly Supervised 3D   
    Classification of Chest CT using Aggregated Multi-Resolution Deep Segmentation Features", 2020 SPIE Medical Imaging: Computer-Aided 
    Diagnosis, Houston, TX, USA. DOI:10.1117/12.2550857
                 


## Network Architecture  
  
  
![Network Architecture](reports/images/network_architecture.png)*Figure 1.  Integrated model architecture for reusing segmentation feature maps in 3D binary classification. The segmentation sub-model is a DenseVNet, taking a variable input volume with a single channel and the classification sub-model is a 3D ResNet, taking an input volume patch of size [112,112,112] with 2 channels. Final output is a tensor with the predicted class probabilities.*  
  
    
    
## Multi-Resolution Deep Segmentation Features  
  
  
![Multi-Resolution Deep Segmentation Features](reports/images/segmentation_features.png)*Figure 2.  From left-to-right: input CT volume (axial view), 3 out of 61 segmentation feature maps extracted from the pretrained DenseVNet model, at different resolutions, and their corresponding static aggregated feature maps (StFA) in the case of diseased lungs with atelectasis (top row), mass (middle row) and emphysema (bottom row).*  
