import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from googlenet_pytorch import GoogLeNet
import matplotlib.pyplot as plt
import numpy
from sklearn import metrics

torch.manual_seed(1)

AlexNet = models.alexnet(pretrained=True)
LeNet = GoogLeNet.from_pretrained('googlenet')
VGG19 = models.vgg.vgg19(pretrained=True)
VGG11 = models.vgg.vgg11(pretrained=True)
ResNet = models.resnet50(pretrained=True)


class ANN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100,50)
        self.fc3 = nn.Linear(50, num_classes)

    def forward(self, x, input_size):
        x = x.view(-1, input_size)  # flatten feature data
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def confusion_graph(actual, predicted):
    conf_graph = metrics.confusion_matrix(actual, predicted)
    cg_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_graph, display_labels = [False, True])