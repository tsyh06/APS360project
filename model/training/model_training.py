import os
import numpy as np
import torch
import time
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

use_cuda = True

# load extracted AlexNet features and labels
train_features = torch.load('.//train_features.pt')
train_labels = torch.load('.//train_labels.pt')
val_features = torch.load('.//val_features.pt')
val_labels = torch.load('.//val_labels.pt')
test_features = torch.load('.//test_features.pt')
test_labels = torch.load('.//test_labels.pt')


# Classifier to train
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


def get_accuracy(model, training=False, test=False):
    if training:
        feature_list = train_features
        label_list = train_labels
    elif test:
        feature_list = test_features
        label_list = test_labels
    else:
        feature_list = val_features
        label_list = val_labels

    correct = 0
    total = 0
    for features, labels in zip(feature_list, label_list):

        #############################################
        # To Enable GPU Usage
        if use_cuda and torch.cuda.is_available():
            features = features.cuda()
            labels = labels.cuda()
        #############################################

        output = model(features)

        # select index with maximum prediction score
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += features.shape[0]
    return correct / total


def train(model, feature_list, label_list, num_epochs=1, lrate=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lrate)

    iters, losses, train_acc, val_acc, epoch_num = [], [], [], [], []

    # training
    n = 0  # the number of iterations
    start_time = time.time()
    for epoch in range(num_epochs):
        mini_b = 0
        for features, labels in zip(feature_list, label_list):

            #############################################
            # To Enable GPU Usage
            if use_cuda and torch.cuda.is_available():
                features = features.cuda()
                labels = labels.cuda()
            #############################################

            out = model(features)  # forward pass
            loss = criterion(out, labels)  # compute the total loss
            loss.backward()  # backward pass (compute parameter updates)
            optimizer.step()  # make the updates for each parameter
            optimizer.zero_grad()  # a cleanup step for PyTorch

            ##### Mini_batch Accuracy #####
            pred = out.max(1, keepdim=True)[1]
            mini_batch_correct = pred.eq(labels.view_as(pred)).sum().item()
            Mini_batch_total = features.shape[0]
            train_acc.append((mini_batch_correct / Mini_batch_total))
            ###########################

            # save the current training information
            iters.append(n)
            losses.append(float(loss) / 64)  # compute *average* loss
            n += 1
            mini_b += 1
            # print("Iteration: ",n, "Time Elapsed: % 6.2f s " % (time.time()-start_time))

        val_acc.append(get_accuracy(model))  # compute validation accuracy
        epoch_num.append(epoch + 1)
        print("Epoch %d Finished. " % epoch, "Time per Epoch: % 6.2f s " % ((time.time() - start_time) / (epoch + 1)))

    torch.save(model.state_dict(), ".//model_save")
    end_time = time.time()

    # plotting
    plt.title("Training Curve")
    plt.plot(iters, losses, label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    plt.title("Training Curve")
    plt.plot(iters, train_acc, label="Training")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

    plt.title("Validation Curve")
    plt.plot(epoch_num, val_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

    train_acc.append(get_accuracy(model, training=True))
    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))
    print("Total time:  % 6.2f s  Time per Epoch: % 6.2f s " % (
        (end_time - start_time), ((end_time - start_time) / num_epochs)))


# training the model
ANN_model = ANNClassifier()
if use_cuda and torch.cuda.is_available():
    ANN_model.cuda()
train(ANN_model, train_features, train_labels, num_epochs=100, lrate=0.0001)


# finding the test accuracy of the model
best_model = ANNClassifier()
if use_cuda and torch.cuda.is_available():
    best_model.cuda()
state = torch.load(".//model_save")
best_model.load_state_dict(state)
accuracy = get_accuracy(best_model, test=True)
print('The best model has test accuracy of: ', accuracy)
