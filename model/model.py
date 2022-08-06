from torchvision import transforms
import torchvision.models as models
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F

AlexNet = models.alexnet(pretrained=True)


# Classifier
class ANNClassifier(nn.Module):
    def __init__(self):
        super(ANNClassifier, self).__init__()
        self.fc1 = nn.Linear(256 * 6 * 6, 200)
        self.fc2 = nn.Linear(200, 74)

    def forward(self, x):
        x = x.view(-1, 256 * 6 * 6)  # flatten feature data
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# return the class of the given image
def image_classification(path='.//test_image.png'):
    model = ANNClassifier()
    state = torch.load(".//final_model", map_location=torch.device('cpu'))
    model.load_state_dict(state)

    classes = ['!', '(', ')', '+', ',', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '=', 'A', 'C', 'Delta',
               'G', 'H', 'M', 'N', 'R', 'S', 'T', 'X', 'alpha', 'b', 'beta', 'cos', 'd', 'div', 'e', 'f', 'forall',
               'gamma', 'geq', 'gt', 'i', 'in', 'infty', 'int', 'j', 'k', 'l', 'lambda', 'leq', 'lim', 'log', 'lt',
               'mu', 'neq', 'o', 'p', 'phi', 'pi', 'pm', 'q', 'sigma', 'sin', 'sqrt', 'sum', 'tan', 'theta', 'times',
               'u', 'v', 'w', 'y', 'z', '{', '}']

    img_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    image = PIL.Image.open(path)
    if path[-3:] == 'jpg':
        image = image.convert('RGB')
    image = img_transform(image)
    features = AlexNet.features(image)
    return classes[model(features).max(1, keepdim=True)[1]]


print(image_classification())
