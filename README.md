# Fabric Classification using Computer Vision

# Repo Contents
The repository contains the entire pipeline for classifying [Intelligent Behaviour Understanding Group (iBUG)](https://ibug.doc.ic.ac.uk/resources/fabrics/) dataset using various texture and surface microgeometry features. 

``` 
.
├── docs                            # Documentation (rubric, proposal)
├── figs                            # Graphs and other visualizations 
├── pkls                            # Pickled data files 
├── subsamples                      # Dataset (.gitignore) 
│   ├── train                       # Train images (80%) 
│   ├── test                        # Test images (20%) 
├── utils                           # Tools and utilities
│   ├── feature_utils.py            # Features and other helper functions 
├── 0_features.ipynb                # Initial feature exploration
|── 0_preprocess.ipynb              # Subdivide and augment images; stratefied train-test split
├── 1_pca.ipynb                     # Principal component analysis including visualization
|── 1_downsample_preprocess.ipynb   # Balance train dataset and resize images to 100x100px
|-- 2_parse_to_vector_down.ipynb    # Parse data to df and add features as vectors in the downsampled version; pickle df
|-- 2_parse_to_vector.ipynb         # Parse data to df and add features as vectors full dataset; pickle df
├── 2_svm_feature_vector.ipynb      # SVM classifier with three features vectorized (normals, HOG, BOVW) in the resized and balanced dataset version
├── 2_model_feature_vector.ipynb    # SVM classifier with three features vectorized (HOG, BOVW) with balanaced dataset
├── 3_parse_to_pickle_scalar.ipynb  # Parse data to df and add features as scalars; pickle df 
├── 3_random_forest.ipynb           # Classifier 1 Cross Validation + Hyper Parameter Tuning - Standard random forest, not accounting for imbalanced classes
├── 3_svm.ipynb                     # Classifier 2 Cross Validation + Hyper Parameter Tuning - Support vector machine
├── 3_RF-BalancedWeighting.ipynb    # Classifier 3 Cross Validation + Hyper Parameter Tuning - Weighted random forest
├── 3_XG-Boost.ipynb                # Classifier 4 Cross Validation + Hyper Parameter Tuning - XGBoost Model w/SMOTE Upsampling
├── 3_Final_models.ipynb            # Summary of All-Star Model Peformance (x4) on Training and Test Sets + Analysis of Computational Cost
├── report.pdf                      # Final report 
└── README.md
``` 
