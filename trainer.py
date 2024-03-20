import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from pathlib import Path
import argparse

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import matplotlib.pyplot as plt
import glob

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


def train(img_siren:Siren, dataloader:DataLoader,hight_res_dataloader:DataLoader, config:TrainConfig)->dict:


    total_steps = config.total_steps # Since the whole image is our dataset, this just means 500 gradient descent steps.
    steps_til_summary = config.steps_til_summary
    lr = config.lr

    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

    losses_agragated = {'train_loss':[], 'test_loss': []}

    for step in range(total_steps):
        loss_no_grad = torch.scalar_tensor(0.).cuda()
        losses_per_image = []
        high_res_loss_no_grad = torch.scalar_tensor(0.).cuda()
        for (model_input, ground_truth), (high_res_model_input, high_res_ground_truth) in zip(dataloader, hight_res_dataloader):
            model_input = model_input.cuda()
            ground_truth = ground_truth.cuda()
            model_output, coords = img_siren(model_input)
            loss = ((model_output - ground_truth)**2).mean()
            with torch.no_grad():
                loss_no_grad += loss
                losses_per_image.append(loss)
                if not step % steps_til_summary:
                    high_res_model_input =  high_res_model_input.cuda()
                    high_res_ground_truth = high_res_ground_truth.cuda()
                    high_res_model_output, high_res_coords = img_siren(high_res_model_input)
                    high_res_loss = ((high_res_model_output - high_res_ground_truth)**2).mean()
                    high_res_loss_no_grad += high_res_loss
                    
            loss.backward()
        optim.step()
        optim.zero_grad()
        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss_no_grad))
            losses_per_image = torch.tensor(losses_per_image)
            losses_agragated['train_loss'].append(loss_no_grad)
            losses_agragated['test_loss'].append(high_res_loss_no_grad)

    losses_agragated['train_loss'] = torch.stack(losses_agragated['train_loss'])
    losses_agragated['test_loss'] = torch.stack(losses_agragated['test_loss'])
    return {'loss': losses_agragated}


def run_exp(train_data_path, high_res_data_path, output_path,images_pairs_names, train_config):
    train_data_path = Path(train_data_path)
    high_res_data_path = Path(high_res_data_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)
    output_vid_path = output_path / 'gt_vs_pred.mp4'
    output_vid_path_high_res =  output_path / 'gt_vs_pred_high_res.mp4'
    output_vid_path_interpulation =  output_path / 'interpulation.mp4'
    output_image_path_interpulation =  output_path / 'interpulation.png'
    convergene_graph =  output_path / 'generalization.png'
    
    plot_output_path = output_path / 'plot.png'
    plot_output_path_high_res = output_path / 'plot_high_res.png'

    hidden_features = train_config.net_params['hidden_features']
    hidden_layers = train_config.net_params['hidden_layers']
    omega_0 = train_config.net_params['omega_0']
    outermost = train_config.net_params['outermost']
    hidden_features = train_config.net_params['hidden_features']
    
    # data loading and traning
    image_dataset = ImageFitting(train_data_path)
    high_res_dataset = ImageFitting(high_res_data_path)
    dataloader = DataLoader(image_dataset, batch_size=1, pin_memory=False, num_workers=0)
    high_res_dataloader = DataLoader(high_res_dataset)
    img_siren = Siren(in_features=image_dataset.coords.shape[-1], out_features=image_dataset.pixels.shape[-1], hidden_features=hidden_features,
                    hidden_layers=hidden_layers, outermost=outermost)
    img_siren.cuda()
    
    train_summery = train(img_siren, dataloader, high_res_dataloader, train_config)
    # show resoults 
    images_tensor = transrom_gt_and_pred_to_a_set_of_contatenated_images(dataloader, img_siren)
    fig, ax  = plt.subplots(figsize=(6,6))
    ax.plot(images_tensor[0][:, 24, 0].detach().cpu(), label='pred')
    ax.plot(images_tensor[0][:, 24+48, 0].detach().cpu(), label='gt')
    ax.legend()
    plt.title('raw 24 in some image, the red channel')
    ax.set_ylabel('red channel intensity')
    ax.set_xlabel('pixel x coord')
    
    plt.savefig(str(plot_output_path))
    plt.close("all")

    vid_creator_compere_gt_to_pard(images_tensor, output_vid_path)
    visualize_network_convergence(train_summery, convergene_graph, train_config)
    high_res_losses = check_image_upsample(high_res_data_path, img_siren, output_vid_path_high_res, plot_output_path_high_res)
    
    interpulation(image_dataset, img_siren, images_pairs_names, train_data_path, out_interp_vid_path = output_vid_path_interpulation, out_interp_img_path = output_image_path_interpulation)
     
    torch.save(img_siren,output_path/ 'siren_model.pth')
    return train_summery


def parse_args():
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(description="Description of your script")

    # Add named arguments
    parser.add_argument("--train_data",default='', help="path to train data")
    parser.add_argument("--high_res_data",default='', help="path to test data")
    parser.add_argument("--output",default='', help="path to experiments artifacts")
    
    # Parse the command-line arguments
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    # input and output pathes
    train_data_path = '/home/yam/workspace/data/cognetive/data/48_test_2'
    high_res_data_path = '/home/yam/workspace/data/cognetive/data/256'
    output_path = '/home/yam/workspace/data/cognetive/data/results'

    args = parse_args()
    train_data_path = args.train_data if args.train_data else train_data_path
    high_res_data_path = args.high_res_data if args.high_res_data else high_res_data_path
    output_path = args.output if args.output else output_path

     
    # pairs for interpolation
    images_pairs_names = [['buy', 'return_purchase'], ['price_tag_euro', 'price_tag_usd'], ['return_purchase','shopping_cart']]

    # nn configuration 
    hidden_features = 512
    hidden_layers = 4
    omega_0 = 30
    outermost = 'linear'
    total_steps = 1000
    steps_til_summary=25
    lr = 1e-4

    # experiment 
    net_architecture = {'hidden_features': hidden_features, 'hidden_layers': hidden_layers, 'omega_0': omega_0, 'outermost': outermost}
    train_config = TrainConfig(total_steps = total_steps, steps_til_summary=steps_til_summary, lr = lr, net_params=net_architecture)
    run_exp(train_data_path, high_res_data_path, output_path,images_pairs_names, train_config)
    print('done!')
