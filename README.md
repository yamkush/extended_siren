# extended_siren

## Installation
1. pip install the following -numpy, matplotlib, torch, cv2 ,os, glob, matplotlib, PIL, torchvision, skimage, pickle, argparse
2. git clone https://github.com/yamkush/extended_siren.git

## Usage
### training Siren
1. Traning:
```
python trainer.py
--train_data <path to train data>
--high_res_data <path to test data>
--output <path to output artifacts>
```
Trainer artifacts - 
1. generalization.png - output graph of the MSE of the test and train data
2. gt_vs_pred.mp4  - video of all estimated images and the train data
3. gt_vs_pred_high_res.mp4 - video of all estimated images w.r.t the test data
4. interpolation.mp4 - interpolation video between two images
5. interpolation.png - interpolation images of all 3 pairs
6. plot.png - a plot of a row from the first images red channel at line 24 for ground trouth and prediction
7. plot_high_res - a plot of a row from the first images red channel at line 124 for ground trouth and prediction from the test data
8. siren_model.pth - saved learned Siren model

In trainer.py in the __main__ function you can play with the nn architercture, and trainer parameters - 
their default is - 
```
hidden_features = 512
hidden_layers = 4
omega_0 = 30
outermost = 'linear'
total_steps = 1000
steps_til_summary=25
lr = 1e-4
```
