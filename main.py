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

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def get_extanded_mgrid(sidelen, dim=2, num_of_images=2):
    tensors =tuple([torch.linspace(-1,1,steps=num_of_images)*2]) + tuple(dim * [torch.linspace(-0.5, 0.5, steps=sidelen)]) 
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim+1)
    mgrid = mgrid[:,[1,2,0]]
    return mgrid

# def get_zifran_mgrid(sidelen, dim=2, num_of_images):
#     mgrid = get_mgrid(sidelen)
#     torch.


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class StepLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()


    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / 30,
                                             np.sqrt(6 / self.in_features) / 30)

    def forward(self, input):
        return self.sigmoid(self.linear(input) - self.linear.bias)*self.linear.bias


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost='linear',
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost=='linear':
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)

        elif outermost=='step_layer':
            self.net.append(StepLayer(hidden_features,out_features, bias=True))
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def get_image_tensor(image_path):
    img = Image.open(image_path)
    
    img2 = Image.fromarray(skimage.data.camera())
    transform = Compose([
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])
    img = transform(img)[:3]
    return img


class ImageFitting(Dataset):
    def __init__(self, sidelength, images_dir:str):
        super().__init__()
        images_path_list = list(glob.glob(str(Path(images_dir)/ '*.png')))
        self.num_of_images = len(images_path_list)
        pixels_list = []
        for image_path in glob.glob(str(Path(images_dir)/ '*.png')):
            img_i = get_image_tensor(image_path)
            pixels_i = img_i.permute(1, 2, 0).view(-1, 3)
            pixels_list.append(pixels_i)
        self.pixels = torch.row_stack(pixels_list)
        self.coords = get_extanded_mgrid(sidelength, 2, self.num_of_images)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0: raise IndexError

        return self.coords, self.pixels

class TrainConfig:
    def __init__(self, total_steps:int, steps_til_summary:int , lr:float):
        self.total_steps = total_steps
        self.steps_til_summary = steps_til_summary
        self.lr = lr

def loss_per_image(sidelen: int, num_of_images: int, pred: torch.Tensor, gt:torch.Tensor) -> list:
    pixels_in_image = sidelen**2
    losses = []
    for i in range(image_dataset.num_of_images):
        losses.append(((pred[0,pixels_in_image*(i):pixels_in_image*(i+1)] - gt[0,pixels_in_image*(i):pixels_in_image*(i+1)])**2).mean())
    return losses


def train(siren:Siren, dataloader:DataLoader, config:TrainConfig)->dict:

    img_siren.cuda()

    total_steps = config.total_steps # Since the whole image is our dataset, this just means 500 gradient descent steps.
    steps_til_summary = config.steps_til_summary
    lr = config.lr

    optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    losses_agragated = []

    for step in range(total_steps):
        model_output, coords = img_siren(model_input)
        loss = ((model_output - ground_truth)**2).mean()

        if not step % steps_til_summary:
            print("Step %d, Total loss %0.6f" % (step, loss))
            # img_grad = gradient(model_output, coords)
            # img_laplacian = laplace(model_output, coords)

            # fig, axes = plt.subplots(1,2, figsize=(18,6))
            # axes[0].imshow(model_output[0,:sidelen**2].cpu().view(sidelen,sidelen,3).detach().numpy())
            # axes[1].imshow(ground_truth[0,:sidelen**2].cpu().view(sidelen,sidelen,3).detach().numpy())
            # plt.show()
            # model_output, coords = img_siren(model_input)
            losses = loss_per_image(sidelen, dataloader.dataset.num_of_images, model_output, ground_truth)
            losses = torch.tensor(losses)
            losses_agragated.append(losses)


        optim.zero_grad()
        loss.backward()
        optim.step()

    return {'losses_vector': losses_agragated}

def visualize_network_convergence(train_summery:dict):
        # plt.figure(0)
    # for lossest in losses_agragated:
    #     plt.plot(torch.log(lossest))
    
    losses_agragated = torch.stack(train_summery['losses_vector'])

    fig, ax =plt.subplots(figsize=(6, 6))

    ax.plot(torch.log(losses_agragated.mean(dim=1)), label='mean')
    ax.plot(torch.log(losses_agragated.max(dim=1)[0]), label='min')
    ax.legend()
    plt.title('images generalization')
    plt.show()

if __name__ == '__main__':
    # configuration 
    images_dir = '/home/yam/workspace/data/cognetive/data/48_test_bigger'
    hidden_features = 256
    hidden_layers = 6
    train_config = TrainConfig(total_steps = 500, steps_til_summary=10, lr = 1e-4)
    sidelen = 48

    # data loading and traning
    image_dataset = ImageFitting(48, images_dir)
    dataloader = DataLoader(image_dataset, batch_size=1, pin_memory=True, num_workers=0)
    img_siren = Siren(in_features=image_dataset.coords.shape[1], out_features=image_dataset.pixels.shape[1], hidden_features=hidden_features,
                    hidden_layers=hidden_layers, outermost='linear')
    train_summery = train(img_siren, dataloader, train_config)
    # show resoults 
    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    visualize_network_convergence(train_summery)

    print('baby')