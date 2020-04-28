
path_file = os.getcwd()
path_file_2 = '/home/ubuntu/.kaggle/images_test_rev1/'
final_path = os.path.join(path_file,path_file_2)

RESIZE_TO = 64
DROPOUT =0.2


def predict(final_path):
    images = []
    x_batch = []
    image = []
    x1 = (424 - 256) // 2
    y1 = (424 - 256) // 2
    for num in os.listdir(final_path):
        if (num.endswith(".jpg")):
            images.append(num)


    for i in images:
        img = cv2.imread("/home/ubuntu/.kaggle/images_test_rev1/" + i)
        img = img[x1:x1 + 256, y1:y1 + 256]
        img = cv2.resize(img, (RESIZE_TO, RESIZE_TO)) / 255
        # image.append(img)
        x_batch.append(img)
    x_batch = np.array(x_batch)
    print(x_batch.shape)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    x = torch.from_numpy(x_batch).float().permute(0, 3, 1, 2)
    print(x.dtype)
    # x = torch.from_numpy(x_batch).float().permute(0, 3, 1, 2)
    # x = torch.FloatTensor(x)

    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, (3, 3))
            self.convnorm1 = nn.BatchNorm2d(16)
            self.pool1 = nn.MaxPool2d((2, 2))
            self.conv2 = nn.Conv2d(16, 32, (3, 3))
            self.convnorm2 = nn.BatchNorm2d(32)
            self.pool2 = nn.AvgPool2d((2, 2))
            self.linear1 = nn.Linear(32 * 14 * 14, 32)
            self.linear1_bn = nn.BatchNorm1d(32)
            self.drop = nn.Dropout(DROPOUT)
            self.linear2 = nn.Linear(32, 37)
            # # self.sigmoid = nn.Sigmoid()
            # self.act = torch.relu
            self.act1 = nn.ReLU()  # nn.LeakyReLu()

        def forward(self, x):
            x = self.pool1(self.convnorm1(self.act1(self.conv1(x))))
            x = self.pool2(self.convnorm2(self.act1(self.conv2(x))))
            # print("output layer size", x.shape)
            x = self.drop(self.linear1_bn(self.act1(self.linear1(x.view(len(x), -1)))))
            x = self.linear2(x)
            return x

    cnn = Net()
    cnn.load_state_dict(torch.load('model_mlgrp1.pt'))

    with torch.no_grad():
        cnn.eval()
        y_pred = cnn(x)

    # print(type(x))
    return torch.sigmoid(y_pred)


y_pred = predict(path_file_2)
print(y_pred.shape)
print(y_pred)
