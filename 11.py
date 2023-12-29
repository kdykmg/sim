import torch
import torch.nn.functional as F
from PIL import Image
import os
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn

path=os.path.dirname(__file__)
path=path+''
os.chdir(path)


class SiameseVGG16(nn.Module):
    def __init__(self):
        super(SiameseVGG16, self).__init__()

        self.vgg16 = models.vgg16(pretrained=True).features
        self.vgg16[0] = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)

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
image1_path = "test/main.jpg"
image1 = Image.open(image1_path)
input1 = preprocess(image1).unsqueeze(0)
output1 = siamese_net.forward_once(input1)

image_distances = []
for i in os.listdir("test/"):
    img_path = os.path.join("test", i)
    image2 = Image.open(img_path)
    input2 = preprocess(image2).unsqueeze(0)
    output2 = siamese_net.forward_once(input2)
    distance = F.pairwise_distance(output1, output2).item()

    image_distances.append((img_path, distance))

sorted_images = sorted(image_distances, key=lambda x: x[1])

print("\nSorted Images:")
for img_path, distance in sorted_images:
    print(f"Image: {img_path}, Distance: {distance}")