EEGMatch: Learning with Incomplete Labels for Semi-Supervised EEG-based Cross-Subject Emotion Recognition
=
* A Pytorch implementation of our under reviewed paper "EEGMatch: Learning with Incomplete Labels for
Semi-Supervised EEG-based Cross-Subject Emotion Recognition".<br> 
* [IEEE TNNLS](https://ieeexplore.ieee.org/document/10756195)
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
# Dataset prepare
* data_prepare_seed.m
# Usage
* After modify setting (path, etc), just run the main function in the implementation_EEGMatch.py
# Acknowledgement
* The implementation code of domain adversarial training is bulit on the [dalib](https://dalib.readthedocs.io/en/latest/index.html) code base 
# Supplementary Materials
* SupplementaryMaterials_EEGMatch.pdf
# Citation
@ARTICLE{10756195,
  author={Zhou, Rushuang and Ye, Weishan and Zhang, Zhiguo and Luo, Yanyang and Zhang, Li and Li, Linling and Huang, Gan and Dong, Yining and Zhang, Yuan-Ting and Liang, Zhen},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={EEGMatch: Learning With Incomplete Labels for Semisupervised EEG-Based Cross-Subject Emotion Recognition}, 
  year={2024},
  volume={},
  number={},
  pages={1-15},
  keywords={Electroencephalography (EEG);emotion recognition;multidomain adaptation;semisupervised learning;transfer learning},
  doi={10.1109/TNNLS.2024.3493425}}

