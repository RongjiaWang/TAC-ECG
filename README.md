# TAC-ECG
This project can be used for pre training and fine-tuning of electrocardiogram signal classification models. The corresponding paper is: Wang R, Dong X, Liu X, et al. TAC-ECG: A task-adaptive classification method for electrocardiogram based on cross-modal contrastive learning and low-rank convolutional adapter [J]. Computer Methods and Programs in Biomedicine, 2025: 108918. 
# Code Description
We have uploaded some code. The subsequent code will be continuously updated.
We used the MIMIC-IV ECG dataset for model pre training. We selected six different ECG signal encoders and conducted fine-tuning experiments on four downstream task datasets. The currently uploaded code takes the Xresnet1d101 model and PTB-XL dataset as examples. In addition, the data architecture corresponding to the code in data preprocessing is consistent with the official website of the dataset. We have not yet addressed issues such as pre training model file paths and dataset file paths, and users need to adapt to their own file organization structure.
# Paper link 
https://www.sciencedirect.com/science/article/abs/pii/S0169260725003359
