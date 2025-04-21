# Progress
# TODO: Change the convolution layers of FYPNet's block AMP and APP to linear layers (completed)
# TODO: Change the optimizer from Adam to AdamW (completed)
# TODO: View the h5 file by typing vitables in the terminal, then open the file (resolved)
# TODO: Modify the learning rate section, there's an issue here (resolved)
# Create a line table to explain this, and finally call poly
# TODO: Modify the dynamic learning rate (completed) then change (completed)
# Poly Strategy, H2Former
# Warmup, after K rounds of training, reduce the lr again
# Spark: https://github.com/keyu-tian/SparK/blob/main/pretrain/utils/lr_control.py, warm-up locked, then reduce lr after K rounds
# Regarding why I gave up on warm-up: warm-up is mainly used for stabilizing large model training in the early stages, and it usually occurs in the initial training phase of the model’s attention module to stabilize the model’s performance
# TODO: Study the encoder part of TransUnet, implement the encoder part, and call a pre-trained model (completed)
# Successfully loaded the Res34 pre-trained model
# TODO: Kvasir and ClinicDB dataset training results are all black, need complete training testing (resolved)
"""
Data Quality Issues: The dataset may have noise, poor image quality, or inaccurate labels, which can lead the model to learn incorrect features or fail to learn the target structure correctly.
Insufficient Data: Too little data
Data Imbalance: The distribution of positive and negative examples in the dataset is imbalanced, causing the model to lean toward predicting the more common category and ignore less frequent ones.
Improper Data Preprocessing: Preprocessing steps such as image size standardization, data augmentation, and image registration can improve the model's robustness and generalization ability.
Improper Hyperparameter Selection: The choice of learning rate, optimizer, loss function, etc., can impact the model's performance.
"""
# Initially believed to be a problem of insufficient model capability
# TODO: Recently found that for segmentation tasks, the edge problem should be focused on, as edge smoothing and edge blurring significantly affect the model's performance (completed)
# Question: For colon polyp data, color information provides some segmentation information, but compared to that, the shape and texture of polyps are more important features for segmentation.
# TODO: Add a validation set, test set, and training set division (completed)
# Successfully divided into train, test, val = 7:2:1
# TODO: Add mode loading method for the dataset (completed)
# isic2018 (completed)
# clinicDB (completed)
# kvasir (completed)
# TODO: FYPNet model has issues, need to fix (completed)
# TODO: Fix the loading of 5 models: unet, bnet, bnet34, duck, unetpp (completed)
# TODO: Use TransUnet's dataset loader to get the dataset (completed)
# TODO: Handle deep supervision by adding feature heads (completed)
# TODO: Fix the test class (completed)