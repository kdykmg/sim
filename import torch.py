import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import os

path=os.path.dirname(__file__)
path=path+''
os.chdir(path)

class SiameseVGG16(nn.Module):
    def __init__(self):
        super(SiameseVGG16, self).__init__()
       
        self.vgg16 = models.vgg16(pretrained=True,).features
        self.vgg16[0] = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2)
    
        self.fc = nn.Sequential(
            nn.Linear(512 * 25 * 25, 4096),  
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 800),
        )

    def forward_once(self, x):
        x = self.vgg16(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


preprocess = transforms.Compose([
    transforms.ToTensor(),

])
siamese_net = SiameseVGG16()

image1_path = "test2/main.jpg"
image1 = Image.open(image1_path)
input1 = preprocess(image1).unsqueeze(0)
for i in os.listdir(path+'/test2/'):
    img=path+'/test2/'+i
    image2 = Image.open(img)
    input2 = preprocess(image2).unsqueeze(0)
    output1, output2 = siamese_net(input1, input2)
    loss1 =  F.pairwise_distance(output1, output2)
    print(img)
    print("loss:", loss1.item())
