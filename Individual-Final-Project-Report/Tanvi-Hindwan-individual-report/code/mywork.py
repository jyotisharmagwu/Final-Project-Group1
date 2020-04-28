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
# %% ===================================================================================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# print(device)
####Data Preprocessing#######
# % =========================================HYPER PARAMETERS============================================================
RESIZE_TO = 64
DROPOUT = 0.2
N_EPOCHS = 100
LR = 0.001
# %% ===================================================================================================================

df = pd.read_csv('/home/ubuntu/training_solutions_rev1.csv')
y = pd.DataFrame(df)
y = y.to_numpy()

# print(y.shape)
# print(y)
# 80,20

# df_train, df_test = train_test_split(df, test_size=.2)
# print(df_train.shape), print(df_test.shape)

ORIG_SHAPE = (424, 424)
CROP_SIZE = (256, 256)
IMG_SHAPE = (64, 64)


def data(dataframe):
    x1 = (424 - 256) // 2
    y1 = (424 - 256) // 2
    sel = dataframe.values
    ids = sel[:, 0].astype(int).astype(str)
    y_batch = sel[:, 1:]
    x_batch = []
    for i in ids:
        image = cv2.imread('/home/ubuntu/images_training_rev1/' + i + '.jpg')

        image = image[x1:x1 + 256, y1:y1 + 256]
        image = cv2.resize(image, IMG_SHAPE) / 255
        x_batch.append(image)
    x_batch = np.array(x_batch)
    return x_batch



x = data(df)
# print(x.shape)
# print(x[1].shape)

# one-hot encode the multi labels using MultiLabelBinarizer()
mlb = MultiLabelBinarizer()
labels = [["Class1.1", "Class1.2", "Class1.3", "Class2.1",
               "Class2.2", "Class3.1", "Class3.2", "Class4.1", "Class4.2",
               "Class5.1", "Class5.2", "Class5.3", "Class5.4", "Class6.1",
               "Class6.2", "Class7.1", "Class7.2", "Class7.3", "Class8.1",
               "Class8.2", "Class8.3", "Class8.4", "Class8.5", "Class8.6",
               "Class8.7", "Class9.1", "Class9.2", "Class9.3", "Class10.1",
               "Class10.2", "Class10.3", "Class11.1", "Class11.2", "Class11.3",
               "Class11.4","Class11.5", "Class11.6"]]

mlb.fit(labels)
y = mlb.transform(y)
# print(y.shape)
# print(y[1:5])
