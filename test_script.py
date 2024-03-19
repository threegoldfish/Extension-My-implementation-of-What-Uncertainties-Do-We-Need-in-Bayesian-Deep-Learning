import os
import matplotlib.pyplot as plt
import time
import argparse

import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import datasets, transforms
import torch.nn.functional as F
import numpy as np



parser = argparse.ArgumentParser(description="Args for test script")
parser.add_argument('--mdl_type', type=str, default='aleatoric', help='Model Type')
args = parser.parse_args()
print(f"Test model type: {args.mdl_type}")

#define model (based on args)
if args.mdl_type == 'aleatoric':
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
                nn.Conv2d(
                    in_channels=1,              # input height
                    out_channels=16,            # n_filters
                    kernel_size=5,              # filter size
                    stride=1,                   # filter movement/step
                    padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
                ),                              # output shape (16, 28, 28)
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 14, 14)
                nn.Dropout(0.5)
            )
            self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
                nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 14, 14)
                nn.ReLU(),
                nn.MaxPool2d(2),                # output shape (32, 7, 7)
                nn.Dropout(0.5)
            )
            # self.linear = nn.Linear(32 * 7 * 7, 10)   # fully connected layer, output 10 classes  [batch_size, 10]
            self.linear = nn.Linear(32 * 7 * 7, CLASS_NUM * 2)

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
            logit = self.linear(x)
            mu, sigma = logit.split(CLASS_NUM, 1)
            return mu, sigma
else:
    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Sequential(    # input shape (1, 28, 28)
                nn.Conv2d(
                    in_channels=1,         # input height
                    out_channels=16,       # n_filters
                    kernel_size=5,         # filter size
                    stride=1,              # filter movement/step
                    padding=2,

                ),                         # output shape (16, 28, 28)
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),      # choose max value in 2x2 area, output shape (16, 14, 14)
                nn.Dropout(0.5)
            )
            self.conv2 = nn.Sequential(           # input shape (16, 14, 14)
                nn.Conv2d(16, 32, 5, 1, 2),       # output shape (32, 14, 14)
                nn.ReLU(),
                nn.MaxPool2d(2),                  # output shape (32, 7, 7)
                nn.Dropout(0.5)
            )
            self.linear = nn.Linear(32 * 7 * 7, CLASS_NUM * 2)  # fully connected layer, output 10 classes  [batch_size, 10]

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x.view(x.size(0), -1)             # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
            logit = self.linear(x)
            mu, sigma = logit.split(CLASS_NUM, 1)
            return mu, sigma
    
#plot digit helper
def plot_digit(first_image, plot_name):
    first_image_np = first_image.numpy()[0]
    plt.figure()
    plt.imshow(first_image_np, cmap='gray')
    plt.title(f'Batch {batch_idx + 1}')
    plt.axis('off')
    plt.savefig(f'./res/mnist/plots/{plot_name}.png')
    plt.show()
    plt.close()

BATCH_SIZE = 100
CLASS_NUM = 10

cnn = CNN()  # Instantiate your model class
saved_state_dict = torch.load(f'./res/mnist/{args.mdl_type}_mdl.pth')
cnn.load_state_dict(saved_state_dict)
cnn.eval()

#define test loader
test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=BATCH_SIZE, shuffle=False)


#test
correct = 0
pred_sigmas = []
pred_correct = []
for batch_idx, (test_x, test_y) in enumerate(test_loader):
    
    test_mu, test_sigma = cnn(test_x)

    pred_y = torch.max(test_mu, 1)[1].data.numpy()
    max_indices = torch.argmax(test_mu, dim=1)
    pred_sigma = test_sigma[torch.arange(max_indices.size(0)), max_indices].detach().numpy()
    pred_sigmas.extend(pred_sigma)
    pred_correct.extend(pred_y == test_y.data.numpy())
    correct += float((pred_y == test_y.data.numpy()).astype(int).sum())

    # Aleatoric uncertainty is measured by some function of test_sigma.


pred_sigmas_sq = np.square(pred_sigmas)
#find index largest sigma_sq
big_10_indices = sorted(range(len(pred_sigmas_sq)), key=lambda i: pred_sigmas_sq[i], reverse=True)[:10]
big_sigma_sq = [pred_sigmas_sq[i] for i in big_10_indices]
big_correct = [pred_correct[i] for i in big_10_indices]
#find index smallest sigma_sq
small_10_indices = sorted(range(len(pred_sigmas_sq)), key=lambda i: pred_sigmas_sq[i])[:10]
small_sigma_sq = [pred_sigmas_sq[i] for i in small_10_indices]
small_correct = [pred_correct[i] for i in small_10_indices]
#find all false prediction and corresponding sigma_sqs
false_indices = [index for index, value in enumerate(pred_correct) if not value]
false_sigma_sq =  [pred_sigmas_sq[i] for i in false_indices]

#compare sigma distribution of false preds and overall preds
x = np.linspace(0, 14, 10)
fig, ax1 = plt.subplots()

ax1.hist(pred_sigmas_sq, bins=30, alpha=0.5, color='blue', label='Overall Sigma_square')
ax1.set_xlabel('Value')
ax1.set_ylabel('Overall Sigma_square Frequency', color='blue')

ax2 = ax1.twinx()
ax2.hist(false_sigma_sq, bins=30, alpha=0.5, color='red', label='False Prediction Sigma_square')
ax2.set_ylabel('False Prediction Sigma_square Frequency', color='red')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
plt.title(f"Variance Comparision between\n Overall Predictions and False Predictions for {args.mdl_type}")
plt.savefig(f'./res/mnist/plots/{args.mdl_type}_dist_comp.png')
        

#plot with biggest sigma_sq
fig, axs = plt.subplots(5, 2, figsize=(10, 30))
counter = 0
img_counter = 0
for batch_idx, (test_x, test_y) in enumerate(test_loader):
    for i, x in enumerate(test_x):
        if counter in big_10_indices:
            x_np = x.numpy()[0]
            row = img_counter // 2
            col = img_counter % 2
            axs[row, col].imshow(x_np, cmap='gray')
            axs[row, col].set_title(f'sigma_square: {pred_sigmas_sq[counter]}')
            axs[row, col].axis('off')
            img_counter += 1
        counter += 1
plt.tight_layout()
fig.suptitle(f'Plots with Biggest Heteroscedastic Uncertainty for {args.mdl_type}', fontsize=25, y=0.99)
plt.savefig(f'./res/mnist/plots/{args.mdl_type}_big.png')
plt.show()

#plot with smallest sigma_sq
fig, axs = plt.subplots(5, 2, figsize=(10, 30))
counter = 0
img_counter = 0
for batch_idx, (test_x, test_y) in enumerate(test_loader):
    for i, x in enumerate(test_x):
        if counter in small_10_indices:
            x_np = x.numpy()[0]
            row = img_counter // 2
            col = img_counter % 2
            axs[row, col].imshow(x_np, cmap='gray')
            axs[row, col].set_title(f'sigma_square: {pred_sigmas_sq[counter]}')
            axs[row, col].axis('off')
            img_counter += 1
        counter += 1
plt.tight_layout()
fig.suptitle(f'Plots with Smallest Heteroscedastic Uncertainty for {args.mdl_type}', fontsize=20, y=0.99)
plt.savefig(f'./res/mnist/plots/{args.mdl_type}_small.png')
plt.show()

accuracy = correct / float(len(test_loader.dataset))
print('test accuracy: %.4f' % accuracy)

#plot train and test statistics
mdls = ['normal', 'aleatoric', 'epistemic', 'combined']
all_loss = []
all_acc = []
for mdl in mdls:
    file_path = "./res/mnist/" + mdl + "_train.txt"
    with open(file_path, 'r') as file:
        all_loss.append([float(x) for x in file.readline().split(",")])
        all_acc.append([float(x) for x in file.readline().split(",")])
        
#training loss        
fig, ax = plt.subplots()
smooth_index = 100
for i, label in enumerate(mdls):
    ax.plot(all_loss[i][::smooth_index], label=label)

ax.set_xlabel('Batch')
ax.set_ylabel('Training Loss (negative log likelihood)')
ax.set_title('Training Losses of Four NNs')
ax.legend()
plt.savefig(f'./res/mnist/plots/train_loss.png')

#test accuracy
fig, ax = plt.subplots()
for i, label in enumerate(mdls):
    ax.plot(all_acc[i], label=label)
ax.set_xlabel('Epoch')
ax.set_ylabel('Test Accuracy')
ax.set_title('Test Accuracy of Four NNs')
ax.legend()
plt.savefig(f'./res/mnist/plots/test_acc.png')
