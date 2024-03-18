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

def get_mgrid(sidelen, dim=2, num_of_images=1):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    mgrid = mgrid.repeat(num_of_images, 1)
    return mgrid

def get_extanded_mgrid(sidelen, dim=2, num_of_images=2):
    tensors =tuple([torch.linspace(-1,1 -2/num_of_images,steps=num_of_images)]) + tuple(dim * [torch.linspace(-1, 1, steps=sidelen)]) 
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim+1)
    mgrid = mgrid[:,[1,2,0]]
    return mgrid

def get_zifran_mgrid(sidelen, dim=2, num_of_images=2):
    R = 1
    mgrid = get_mgrid(sidelen, dim, num_of_images)
    mgrid2 = - torch.tensor([1, 1]) - mgrid
    mgrid3 = torch.stack((mgrid.norm(dim=1), torch.atan2(mgrid[:,0], mgrid[:,1])),dim=1)
    mgrid_z = get_extanded_mgrid(sidelen, dim, num_of_images)[:, -1][:,None]
    mgrid4 = R * torch.sin(mgrid_z*np.pi)
    mgrid5 = R * torch.cos(mgrid_z*np.pi)
    mgrid5 = torch.cat((mgrid, mgrid2, mgrid4, mgrid5), dim=1)
    mgrid6 = mgrid5.reshape(num_of_images, sidelen**2, -1)
    return mgrid6


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
        x = self.linear(input)
        # omega_vec.shape = torch.linspace(0,256, x.shape[-1])[None,None]
        return torch.cos(self.omega_0 * self.linear(input))

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

    img_tensor = ToTensor()(img)
    mask = (img_tensor[3] == 0)
    img_tensor = img_tensor.permute(1,2,0)
    img_tensor[mask, :] = 1
    img_tensor = img_tensor.permute(2,0,1)
    img_out = Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))(img_tensor)

    # transform = Compose([
    #     ToTensor(),
    #     Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    # ])
    # img = transform(img)
    # img2 = img
    # img2[:,img2[3] == 0] = [0,0,0]
    # img2 = img2[:3]
    return img_out[:3]


class ImageFitting(Dataset):
    def __init__(self, sidelength, images_dir:str):
        super().__init__()
        images_path_list = sorted(glob.glob(str(Path(images_dir)/ '*.png')))
        self.num_of_images = len(images_path_list)
        pixels_list = []
        for image_path in images_path_list:
            img_i = get_image_tensor(image_path)
            pixels_i = img_i.permute(1, 2, 0).view(-1, 3)
            pixels_list.append(pixels_i)
        self.pixels = torch.stack(pixels_list)
        self.coords = get_zifran_mgrid(sidelength, 2, self.num_of_images)

    def __len__(self):
        return self.coords.shape[0]

    def __getitem__(self, idx):
        if idx > self.coords.shape[0]: raise IndexError

        return self.coords[idx], self.pixels[idx]

    