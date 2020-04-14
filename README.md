# Prostate Detection in mpMRI using 2D RetinaNet
 
**Problem Statement**: Localization of the prostate (healthy or with benign/malignant tumors) in multi-parametric MRI scans (T2W, DWI *with high b-value*, ADC).   

**Data**: 1950 prostate mpMRI volumes (*Healthy/Benign Cases*: 1234; *Malignant Cases*: 716); equivalent to 23400 2D slices. [1559/391: Train/Val Ratio] 

**Acknowledgments**: The following approach is based on a TensorFlow Estimator/Keras (v1.15) adaptation of [keras-retinanet](https://github.com/fizyr/keras-retinanet/) by Fizyr, [SEResNet](https://github.com/qubvel/classification_models) by Pavel Yakubovskiy et al., and an [anchor optimization algorithm](https://github.com/martinzlocha/anchor-optimization) by Martin Zlocha et al.

**Directories**  
  ● Preprocess Dataset to Normalized Volumes in Optimized NumPy Format: `scripts/preprocess.py`  
  ● Generate Data-Directory Feeder List: `scripts/feeder_csv.py`  
  ● Anchor Optimization: `misc/rdc_08.py`  
  ● Pre-Calculate Regression Target Deltas *(to determine Mean, STDEV): `misc/rdc_07.py`  
  ● Train 2D RetinaNet Model: `scripts/train_RetinaNet.py`  
  ● Deploy Model (Validation): `scripts/deploy_model.py`  
  

**Reference Publications:**  
  ● Tsung-Yi Lin et al. (2017), "Focal Loss for Dense Object Detection", IEEE ICCV. DOI:10.1109/ICCV.2017.324  
  ● M. Zlocha et al. (2019), "Improving RetinaNet for CT Lesion Detection with Dense Masks from Weak RECIST Labels", MICCAI. DOI:10.1007/978-3-030-32226-7_45                 


## Network Architecture  
  
  
![Network Architecture](reports/images/network_architecture.png)*Figure 1.  Integrated model architecture for reusing segmentation feature maps in 3D binary classification. The segmentation sub-model is a DenseVNet, taking a variable input volume with a single channel and the classification sub-model is a 3D ResNet, taking an input volume patch of size [112,112,112] with 2 channels. Final output is a tensor with the predicted class probabilities.*  
  
    
    
## Multi-Resolution Deep Segmentation Features  
  
  
![Multi-Resolution Deep Segmentation Features](reports/images/segmentation_features.png)*Figure 2.  From left-to-right: input CT volume (axial view), 3 out of 61 segmentation feature maps extracted from the pretrained DenseVNet model, at different resolutions, and their corresponding static aggregated feature maps (StFA) in the case of diseased lungs with atelectasis (top row), mass (middle row) and emphysema (bottom row).*  
