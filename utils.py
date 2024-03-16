import cv2
import os
import glob
from pathlib import Path
import torch

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

def transrom_gt_and_pred_to_a_set_of_contatenated_images(ground_truth,model_output, num_of_images, sidelen):
    images_list = []
    num_of_pixels = sidelen**2
    for n in range(num_of_images):
        im_gt = ground_truth[0][n*num_of_pixels:(n+1)*num_of_pixels].cpu().view(sidelen,sidelen,3).detach().numpy()
        im_pred = model_output[0][n*num_of_pixels:(n+1)*num_of_pixels].cpu().view(sidelen,sidelen,3).detach().numpy()
        caption = f"GT Frame: {n}"
        cv2.putText(im_gt, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        caption = f"pred Frame: {n}"
        cv2.putText(im_pred, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        image = torch.cat([torch.tensor(im_pred), torch.tensor(im_gt)], dim=1)
        image = image.mul(255).clip(0,255).to(torch.uint8)
        images_list.append(image)

    return torch.stack(images_list)
        

   