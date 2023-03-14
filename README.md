EEGMatch: Learning with Incomplete Labels for Semi-Supervised EEG-based Cross-Subject Emotion Recognition
=
* A Pytorch implementation of our paper "EEGMatch: Learning with Incomplete Labels for
Semi-Supervised EEG-based Cross-Subject Emotion Recognition".<br> 

# Installation:
* Python 3.7
* Pytorch 1.3.1
* NVIDIA CUDA 9.2
* Numpy 1.20.3
* Scikit-learn 0.23.2
* scipy 1.3.1

# Preliminaries
* Prepare dataset: [SEED](https://bcmi.sjtu.edu.cn/~seed/index.html) and [SEED-IV](https://bcmi.sjtu.edu.cn/~seed/index.html)

# Training 
* EEGMatch model definition file: model_EEGMatch.py 
* Pipeline of the EEGMatch: implementation_EEGMatch.py
* implementation of domain adversarial training: Adversarial_DG.py
# Usage
* After modify setting (path, etc), just run the main function in the implementation_EEGMatch.py
# Acknowledgement
* The implementation code of domain adversarial training is bulit on the [dalib](https://dalib.readthedocs.io/en/latest/index.html) code base 

