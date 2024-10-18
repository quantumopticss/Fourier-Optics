"""
in this code we will use a pretrained alexnet to do image classification
"""
#%%
import torch as th
from torchvision import models
from torchvision import transforms

#%%
resnet = models.resnet101()
preprocess = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                    )])
#%%
from PIL import Image
img = Image.open("./cat.png")
img_t = preprocess(img)
batch_t = th.unsqueeze(img_t, 0)

resnet.eval()
out = resnet(batch_t)
    
_, index = th.max(out, 1)
percentage = th.nn.functional.softmax(out, dim=1)[0] * 100
print(percentage)
