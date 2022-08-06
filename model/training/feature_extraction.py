import torch
from torchvision import datasets, models, transforms
torch.manual_seed(1)

AlexNet = models.alexnet(pretrained=True)

dataset_dir = '.\\dataset'

data_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
data = datasets.ImageFolder(dataset_dir, transform=data_transform)

# split data into train/val/test in ratio 60/20/20
train, val, test = torch.utils.data.random_split(data, [35520, 11840, 11840])

train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=True)


############################################################################
# extract training features
dataiter = iter(train_loader)

features = []
labels = []
for images, label in dataiter:
    features.append(AlexNet.features(images))
    labels.append(label)

torch.save(features, '.\\train_features.pt')
torch.save(labels, '.\\train_labels.pt')

print('Training Features and Labels Saved')
del features, labels


############################################################################
# extract validation features
dataiter = iter(val_loader)

features = []
labels = []
for images, label in dataiter:
    features.append(AlexNet.features(images))
    labels.append(label)

torch.save(features, '.\\val_features.pt')
torch.save(labels, '.\\val_labels.pt')

print('Validation Features and Labels Saved')
del features, labels


#############################################################################
# extract test features
dataiter = iter(test_loader)

features = []
labels = []
for images, label in dataiter:
    features.append(AlexNet.features(images))
    labels.append(label)

torch.save(features, '.\\test_features.pt')
torch.save(labels, '.\\test_labels.pt')

print('Test Features and Labels Saved')
del features, labels
