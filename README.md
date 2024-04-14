# Fabric Classification using Computer Vision

# Repo Contents
The repository contains the entire pipeline for classifying [Intelligent Behaviour Understanding Group (iBUG)](https://ibug.doc.ic.ac.uk/resources/fabrics/) dataset using various texture and surface microgeometry features. 

.
├── docs                     # Documentation (rubric, proposal)
├── figs                     # Graphs and other visualizations 
├── pkls                     # Pickled data files 
├── subsamples               # Dataset (.gitignore) 
│   ├── train                # Train images (80%) 
│   ├── test                 # Test images (20%) 
├── utils                    # Tools and utilities
│   ├── feature_utils.py     # Features and other helper functions 
├── features.ipynb           # Initial feature exploration
├── parse_to_pickle.ipynb    # Parse data to df and add features; pickle df 
├── pca.ipynb                # Principal component analysis including visualization
├── preprocess.ipynb         # Subdivide and augment images; stratefied train-test split
├── random_forest.ipynb      # Classifier 1 
├── svm.ipynb                # Classifier 2
├── report.pdf               # Final report 
└── README.md
