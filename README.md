# Layer 2 of PROCESS Use Case 1 application

This repository contains the first Use Case Application for UC#1 of the PROCESS project, http://www.process-project.eu/, 
in particular the second layer of the software architecture CamNet. 

# UC1_medicalImaging
The use case tackles cancer detection and tissue classification on the latest challenges in cancer research using histopathology images from CAMELYON 16 and 17. 

# CAMNET: a three-layered software architecture
The software implemented by the use case consists of three layers. 
L 1. Data extraction and preprocessing: https://github.com/medgift/PROCESS_L1
L 2. Network training
L 3. Network interpretability 

# Dependencies
The code is written in Python 2.7 and requires Keras 2.1.5 with Tensorflow 1.4.0 as backend. Further dependencies are in requirements.txt.

# Configuration
Configuration files are ini-based. A full template is in doc/config.cfg.

# Usage

The master script is a pipeline-based program that can be run by the command

python train_cnn.py GPU_DEVICE EXPERIMENT_NAME RANDOM_SEED

GPU_DEVICE = GPU index on server
EXPERIMENT_NAME = name of the experiment
RANDOM_SEED = seed for reproducibility. 

For example, you can run the script as follows: 

python train_cnn.py 0 debug_run 1001


