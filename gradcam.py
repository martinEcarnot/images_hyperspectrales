import numpy as np
import cv2
import torch
from torchvision import transforms
from torch.nn import functional as F
import spectral as sp


# define computation device
device = ('cuda' if torch.cuda.is_available() else 'cpu')
# initialize model, switch to eval model, load trained weights
model = torch.load('models/CNN_2_2/CNN_2_2.pth')
#model.load_state_dict(model)



def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def show_cam(CAMs, width, height, orig_image, class_idx, save_name):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.5 + orig_image * 0.5
        # put class label text on the result
        cv2.putText(result, str(int(class_idx[i])), (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('CAM', result/255.)
        
        

features_blobs = []
def hook_feature(output):
    features_blobs.append(output.data.cpu().numpy())
    
    
model._modules.get('conv3').register_forward_hook(hook_feature)
# get the softmax weight
params = list(model.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())

# define the transforms, resize => tensor => normalize
transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((28, 28)),
     transforms.ToTensor(),
     transforms.Normalize(
        mean=[0.5],
        std=[0.5])
    ])

# run for all the images in the `input` folder
image_path = 'img/cropped/RGB/var1_2020_x75y20_8000_us_2x_2022-04-26T122543_corr_grain0.hdr'
    # read the image
image = sp.open_image(image_path)
print(image[0])
print(image.shape)
image = np.array(image)
print(image.shape)
orig_image = image.copy()
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = np.expand_dims(image, axis=2)
height, width, _ = orig_image.shape
    # apply the image transforms
image_tensor = transform(image)
    # add batch dimension
image_tensor = image_tensor.unsqueeze(0)
    # forward pass through model
probs = model(image_tensor)
    # get the softmax probabilities

class_idx = [probs.softmax(1)]
    
    # generate class activation mapping for the top1 prediction
CAMs = returnCAM(features_blobs[0], weight_softmax, class_idx)
    # file name to save the resulting CAM image with
save_name = f"{image_path.split('/')[-1].split('.')[0]}"
    # show and save the results
show_cam(CAMs, width, height, orig_image, class_idx, save_name)