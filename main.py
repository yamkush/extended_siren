import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from pathlib import Path

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
import glob

import time

from utils import vid_creator_compere_gt_to_pard, transrom_gt_and_pred_to_a_set_of_contatenated_images, interpulation, check_image_upsample, visualize_network_convergence
from siren import *


class TrainConfig:
    def __init__(self, total_steps:int, steps_til_summary:int , lr:float, net_params: dict):
        self.total_steps = total_steps
        self.steps_til_summary = steps_til_summary
        self.lr = lr
        self.net_params = net_params

def loss_per_image(sidelen: int, num_of_images: int, pred: torch.Tensor, gt:torch.Tensor) -> list:
    pixels_in_image = sidelen**2
    losses = []
    for i in range(image_dataset.num_of_images):
        losses.append(((pred[0,pixels_in_image*(i):pixels_in_image*(i+1)] - gt[0,pixels_in_image*(i):pixels_in_image*(i+1)])**2).mean())
    return losses


def train(img_siren:Siren, dataloader:DataLoader, config:TrainConfig)->dict:


    total_steps = config.total_steps # Since the whole image is our dataset, this just means 500 gradient descent steps.
    steps_til_summary = config.steps_til_summary
    lr = config.lr

    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

    losses_agragated = []

    for step in range(total_steps):
        loss_no_grad = torch.scalar_tensor(0.).cuda()
        losses_per_image = []
        for model_input, ground_truth in dataloader:
            model_input = model_input.cuda()
            ground_truth = ground_truth.cuda()
            model_output, coords = img_siren(model_input)
            loss = ((model_output - ground_truth)**2).mean()
            with torch.no_grad():
                loss_no_grad += loss
                losses_per_image.append(loss)
            loss.backward()
        optim.step()
        optim.zero_grad()
        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss_no_grad))
            losses_per_image = torch.tensor(losses_per_image)
            losses_agragated.append(losses_per_image)
    return {'losses_vector': losses_agragated}


def vid_creator(images):

    image_files = sorted(os.listdir(image_dir))

    # Define the output video file name
    output_video = "output_video.avi"

    # Get the first image to extract dimensions
    first_image = cv2.imread(os.path.join(image_dir, image_files[0]))
    height, width, _ = first_image.shape

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video, fourcc, 10.0, (width, height))

    # Loop through each image and add it to the video
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        frame = cv2.imread(image_path)
        out.write(frame)

    # Release the VideoWriter and close all OpenCV windows
    out.release()
    cv2.destroyAllWindows()

def run_exp(train_data_path, high_res_data_path, output_path, train_config):
    # configuration 
    # main_dir = Path('/home/yam/workspace/data/cognetive/data/')
    # output_path = input_path / 'results'
    output_path.mkdir(exist_ok=True)
    # train_data_path = input_path / '48_test'
    # high_res_train_data_path = input_path/ '256_test'
    output_vid_path = output_path / 'gt_vs_pred.mp4'
    output_vid_path_high_res =  output_path / 'gt_vs_pred_high_res.mp4'
    output_vid_path_interpulation =  output_path / 'interpulation.mp4'
    output_image_path_interpulation =  output_path / 'interpulation.png'
    convergene_graph =  output_path / 'generalization.png'
    images_pairs_names = [['buy', 'return_purchase'], ['price_tag_euro', 'price_tag_usd'], ['return_purchase','shopping_cart']]
    plot_output_path = output_path / 'plot.png'
    plot_output_path_high_res = output_path / 'plot_high_res.png'

    hidden_features = train_config.net_params['hidden_features']
    hidden_layers = train_config.net_params['hidden_layers']
    omega_0 = train_config.net_params['omega_0']
    outermost = train_config.net_params['outermost']
    # net_architecture = {'hidden_features': hidden_features, 'hidden_layers': hidden_layers, 'omega_0': omega_0, 'outermost': 'linear'}
    # train_config = TrainConfig(total_steps = 1000, steps_til_summary=10, lr = 1e-4, net_params=net_architecture)
    hidden_features = train_config.net_params['hidden_features']
    sidelen = 48
    sidelen_highres = 256

    # data loading and traning
    image_dataset = ImageFitting(48, train_data_path)
    dataloader = DataLoader(image_dataset, batch_size=1, pin_memory=False, num_workers=0)
    img_siren = Siren(in_features=image_dataset.coords.shape[-1], out_features=image_dataset.pixels.shape[-1], hidden_features=hidden_features,
                    hidden_layers=hidden_layers, outermost=outermost)
    img_siren.cuda()
    # train_summery['upsample_losses'] = check_image_upsample(high_res_images_dir, sidelen_highres, img_siren)

    train_summery = train(img_siren, dataloader, train_config)
    # show resoults 
    images_tensor = transrom_gt_and_pred_to_a_set_of_contatenated_images(dataloader, img_siren, sidelen)
    ax  = plt.subplot(111)
    ax.plot(images_tensor[0][:, 24, 0].detach().cpu())
    ax.plot(images_tensor[0][:, 24+48, 0].detach().cpu())
    plt.title('raw 24 in some image, the red channel')
    plt.savefig(str(plot_output_path))
    plt.show()
    plt.imshow(images_tensor[0].detach().cpu())
    plt.show()

    vid_creator_compere_gt_to_pard(images_tensor, output_vid_path)
    visualize_network_convergence(train_summery, convergene_graph, train_config)
    check_image_upsample(high_res_data_path, sidelen_highres, img_siren, output_vid_path_high_res, plot_output_path_high_res)
    
    interpulation(image_dataset, img_siren, images_pairs_names, train_data_path, sidelen= sidelen, out_interp_vid_path = output_vid_path_interpulation, out_interp_img_path = output_image_path_interpulation)
    



if __name__ == '__main__':
    data_dir = Path('/home/yam/workspace/data/cognetive/data/')
    train_data_path = data_dir / '48_test'
    high_res_data_path = data_dir/ '256_test'
    output_path = data_dir / 'results'

    hidden_features = 256
    hidden_layers = 2
    omega_0 = 30
    outermost = 'linear'
    total_steps = 60
    steps_til_summary=10
    lr = 1e-4

    net_architecture = {'hidden_features': hidden_features, 'hidden_layers': hidden_layers, 'omega_0': omega_0, 'outermost': outermost}
    train_config = TrainConfig(total_steps = total_steps, steps_til_summary=steps_til_summary, lr = lr, net_params=net_architecture)
    run_exp(train_data_path, high_res_data_path, output_path, train_config)
    print('done!')
