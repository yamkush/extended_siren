import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import torch

from pathlib import Path
from torch.utils.data import DataLoader, Dataset

from siren import ImageFitting

def vid_creator_compere_gt_to_pard(images, output_video):
    
    
    # Get the first image to extract dimensions
    first_image =images[0]
    height, width, _ = first_image.shape

    # Define the video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for MP4 format
    out = cv2.VideoWriter(str(output_video), fourcc, 5.0, (width, height))

    # Loop through each image and add it to the video
    for frame in images:
        cv2.cvtColor(frame.numpy(), cv2.COLOR_BGR2RGB)
        out.write(cv2.cvtColor(frame.numpy(), cv2.COLOR_BGR2RGB))

    # Release the VideoWriter and close all OpenCV windows
    out.release()
    cv2.destroyAllWindows()

def transrom_gt_and_pred_to_a_set_of_contatenated_images(dataloader: DataLoader,img_siren):
    sidelen = dataloader.dataset.sidelen
    with torch.no_grad():
        images_list = []
        num_of_pixels = sidelen**2
        for model_input, ground_truth in dataloader:
            model_output, coords = img_siren(model_input.cuda())
            im_gt = ground_truth[0].cpu().view(sidelen,sidelen,3).detach().numpy()
            im_pred = model_output[0].cpu().view(sidelen,sidelen,3).detach().numpy()
            caption = f"GT"
            cv2.putText(im_gt, caption, (1, 3), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 255, 255),1)
            caption = f"pr"
            cv2.putText(im_pred, caption, (1, 3), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (255, 255, 255), 1)
            image = torch.cat([torch.tensor(im_pred), torch.tensor(im_gt)], dim=1)
            image = image.clip(0,1).mul(255).to(torch.uint8)
            images_list.append(image)

    return torch.stack(images_list)
        
def interpulation(dataset, img_siren, images_pairs_names, images_dir, out_interp_vid_path, out_interp_img_path):
    images_path_list = sorted(glob.glob(str(Path(images_dir)/ '*.png')))
    sidelen = dataset.sidelen
    images_names = [Path(im_path).stem for im_path in images_path_list]
    images_list = []
    for pair_names in images_pairs_names:
        idx1 = [index for index, name in enumerate(images_names) if pair_names[0] in name]
        idx2 = [index for index, name in enumerate(images_names) if pair_names[1] in name]
        
        if len(idx1) >= 1 and len(idx2) >= 1:
            idx1 = idx1[0]
            idx2 = idx2[0]
            images_sub_list = []
            model_input1, _ = dataset[idx1]
            model_input2, _ = dataset[idx2]
            alphas = torch.linspace(0, 1, 15)
            # Define the video codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for MP4 format
            out = cv2.VideoWriter(str(out_interp_vid_path), fourcc, 5.0, (sidelen, sidelen))
            
            for alpha in alphas:
                model_input_alpha = model_input1*alpha + model_input2 * (1-alpha)
                model_output, _ =  img_siren(model_input_alpha.cuda())
                img = model_output.view(sidelen,sidelen,3).clip(0,1).mul(255).to(torch.uint8)
                images_sub_list.append(img)
                out.write(cv2.cvtColor(img.detach().cpu().numpy(), cv2.COLOR_BGR2RGB))
            # Release the VideoWriter and close all OpenCV windows
            out.release()
            cv2.destroyAllWindows()
            images_list.append(images_sub_list)
        
        horez_concat_images_list = []
        if len(images_list) == 0:
            return 
            
        for image in images_list:
            horez_concat_images_list.append(torch.cat(image, dim=1))

        images_matrix = torch.cat(horez_concat_images_list, dim=0)
        cv2.imwrite(str(out_interp_img_path), images_matrix.cpu().numpy() )
         
def check_image_upsample(images_dir:str, img_siren, output_path, plot_output_path):

    # data loading and traning
    image_dataset = ImageFitting( images_dir)
    sidelen = image_dataset.sidelen
    image_dataset.pixels = image_dataset.pixels.cpu()
    image_dataset.coords = image_dataset.coords.cpu()
    dataloader = DataLoader(image_dataset, pin_memory=True, batch_size=1, num_workers=0)
    images_tensor = transrom_gt_and_pred_to_a_set_of_contatenated_images(dataloader, img_siren)
    vid_creator_compere_gt_to_pard(images_tensor, output_path)
    losses = []
    first_image_flag = True
    with torch.no_grad():
        for model_input, ground_truth in dataloader:
            model_input = model_input.cuda()
            model_output, coords = img_siren(model_input)
            if first_image_flag:
                pred_im = model_output[0].view(sidelen, sidelen, 3).mul(255).clip(0,255)
                gt_im = ground_truth[0].view(sidelen,sidelen, 3).mul(255).clip(0,255)
                ax  = plt.subplot(111)
                ax.plot(pred_im[:, 128, 0].detach().cpu(), label='pred')
                ax.plot(gt_im[:, 128, 0].detach().cpu(), label= 'gt')
                ax.legend()
                ax.set_ylabel('red channel intensity')
                ax.set_xlabel('pixel x coord')
                plt.title('raw 128 in some image, the red channel')
                plt.savefig(str(plot_output_path))
                
                plt.close("all")
                first_image_flag = False

            losses.append(((model_output - ground_truth.cuda())[0]**2).mean(dim=1).mean())
    return torch.stack(losses)

def visualize_network_convergence(train_summery:dict, output_path: Path, train_config):

    train_loss = train_summery['loss']['train_loss'].detach().cpu()
    test_loss = train_summery['loss']['test_loss'].detach().cpu()
    x_axis = np.arange(train_loss.shape[0])*train_config.steps_til_summary
    fig, ax =plt.subplots(figsize=(6, 6))
    ax.plot(x_axis, torch.log(train_loss), label='train')
    ax.plot(x_axis, torch.log(test_loss), label='test')
    ax.legend()
    plt.title('images generalization')
    plt.xlabel('iteration')
    plt.ylabel('log error')
    plt.savefig(str(output_path))
    plt.close("all")
