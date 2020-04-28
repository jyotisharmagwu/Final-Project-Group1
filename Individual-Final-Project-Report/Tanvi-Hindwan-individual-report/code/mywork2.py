import cv2
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from skimage.transform import resize
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import cv2
import warnings
warnings.filterwarnings("ignore")
import os
import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from torch.autograd import Variable
import warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
import torch.utils.data as utils
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, BCELoss, BCEWithLogitsLoss
from torch.optim import Adam, SGD
from sklearn.metrics import accuracy_score

import torchsample
#####Intializing the data loader here #######



transform = torchsample.transforms.Compose([
     torchsample.transforms.Rotate(90),
     torchsample.transforms.RandomFlip()
     #torchsample.transforms.Translate(0.04)
                # translation decreases performance!
])

trainset = torchsample.TensorDataset(
    (x_train), (y_train),
     input_transform = transform
 )


trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=1024, shuffle=True, num_workers=0
)

evalset = torch.utils.data.TensorDataset(
    (x_test), (y_test)
)

evalloader = torch.utils.data.DataLoader(
    evalset, batch_size=1024, shuffle=False, num_workers=0
)








print(x_train.shape)


