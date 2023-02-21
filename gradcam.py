import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models
from skimage.io import imread
from skimage.transform import resize
import spectral as sp

class GradCamModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradients = None
        self.tensorhook = []
        self.layerhook = []
        self.selected_out = None
        
        #PRETRAINED MODEL
        self.pretrained = torch.load('models/CNN_2_2/CNN_2_2.pth')
        #self.layerhook.append(self.pretrained.layer4.register_forward_hook(self.forward_hook()))
        
        for p in self.pretrained.parameters():
            p.requires_grad = True
    
    def activations_hook(self,grad):
        self.gradients = grad

    def get_act_grads(self):
        return self.gradients

    def forward_hook(self):
        def hook(module, inp, out):
            self.selected_out = out
            self.tensorhook.append(out.register_hook(self.activations_hook))
        return hook

    def forward(self,x):
        out = self.pretrained(x)
        return out, self.selected_out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gcmodel = GradCamModel().to(device)


img = sp.open_image('img/cropped/RGB/' + 'var1_2020_x75y20_8000_us_2x_2022-04-26T122543_corr_grain2' + '.hdr')
img_tensor = torch.tensor(img[:, :, :])

    
inpimg = img_tensor.to(device, torch.float32)
inpimg = torch.permute(inpimg, (2, 0, 1)).float()


out, acts = gcmodel(inpimg)
acts = acts.detach().cpu()
loss = nn.CrossEntropyLoss()(out,torch.from_numpy(np.array([600])).to(device))
loss.backward()
grads = gcmodel.get_act_grads().detach().cpu()
pooled_grads = torch.mean(grads, dim=[0,2,3]).detach().cpu()
for i in range(acts.shape[1]):
    acts[:,i,:,:] += pooled_grads[i]
heatmap_j = torch.mean(acts, dim = 1).squeeze()
heatmap_j_max = heatmap_j.max(axis = 0)[0]
heatmap_j /= heatmap_j_max

heatmap_j = resize(heatmap_j,(224,224),preserve_range=True)

cmap = mpl.cm.get_cmap('jet',256)
heatmap_j2 = cmap(heatmap_j,alpha = 0.2)

fig, axs = plt.subplots(1,1,figsize = (5,5))
axs.imshow((img*std+mean)[0].transpose(1,2,0))
axs.imshow(heatmap_j2)
plt.show()

#other visualisation type
heatmap_j3 = (heatmap_j > 0.75)

fig, axs = plt.subplots(1,1,figsize = (5,5))
axs.imshow(((img*std+mean)[0].transpose(1,2,0))*heatmap_j3)
plt.show()