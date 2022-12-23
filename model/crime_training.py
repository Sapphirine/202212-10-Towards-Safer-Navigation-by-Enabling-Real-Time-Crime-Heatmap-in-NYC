import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np

import numpy as np
from sklearn.linear_model import LinearRegression

# raw = pd.read_csv("dataset_encode_balance.csv").values


'''
raw = pd.read_csv("dataset_encode_balance.csv").values
mean_val_0 = np.mean(raw[:,1])
std_val_0 = np.std(raw[:,1])
mean_val_1 = np.mean(raw[:,2])
std_val_1 = np.std(raw[:,2])
data = np.zeros((raw.shape[0], raw.shape[1]), dtype = np.float64)
for i in range(raw.shape[0]):
    data[i,0] = (int(raw[i][0].split("/")[0])-1)/11
    time = raw[i][0].split(" ")[1].split(":")
    data[i,1] = (int(time[0])*60+int(time[1]))/3600
    data[i,2] = (raw[i,1]-mean_val_0)/std_val_0
    data[i,3] = (raw[i,2]-mean_val_1)/std_val_1

data[:,4] = raw[:,3]
data[:,5] = raw[:,4]
data[:,6] = raw[:,5]

x_data = data
y_data = raw[:, -1]*3

print(type(x_data))
print(x_data[0])

model = LinearRegression()
model.fit(x_data, y_data)

x_test = data[0:10]

predict = model.predict(x_test)

print(predict)
print(y_data[0:10])

from sklearn.externals import joblib
joblib.dump(model, 'lr.model')
# lr = joblib.load('lr.model')
'''

test = pd.read_csv("dataset_encode.csv").values
mean_val_0 = np.mean(test[0:1085501,1])
std_val_0 = np.std(test[0:1085501,1])
mean_val_1 = np.mean(test[0:1085501,2])
std_val_1 = np.std(test[0:1085501,2])

class myDataSet(Dataset):
    def __init__(self, data_dir):
        self.raw = pd.read_csv(data_dir).values
 
    def __len__(self):
        return len(self.raw)
 
    def __getitem__(self, index):
        
        data_list = []

        date = self.raw[index][0]
        data_list.append((int(date.split("/")[0])-1)/11)
        time = date.split(" ")[1].split(":")
        data_list.append((int(time[0])*60+int(time[1]))/3600)
 
        data_list.append((self.raw[index][1]-mean_val_0)/std_val_0)
        data_list.append((self.raw[index][2]-mean_val_1)/std_val_1)
        data_list.append(self.raw[index][3]/4)
        data_list.append(self.raw[index][4]/5)
        data_list.append(self.raw[index][5])

        data = torch.tensor(np.array(data_list, dtype=np.float32))
        label = torch.tensor(self.raw[index][6])
     
        return data, label

class crime_predict(nn.Module):
    def __init__(self):
        super(crime_predict, self).__init__()
        self.linear1=torch.nn.Linear(7,512)
        self.relu=torch.nn.ReLU()
        self.linear2=torch.nn.Linear(512,512)
        self.relu2=torch.nn.ReLU()
        self.linear3=torch.nn.Linear(512,512)
        self.relu3=torch.nn.ReLU()
        self.linear4=torch.nn.Linear(512,4)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        return x

training_data = myDataSet("dataset_encode.csv")

train_dataloader = DataLoader(training_data, batch_size=512, shuffle=True)

net = crime_predict()
print(net)
net.cuda()

optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

for epoch in range(5):
    for batch_idx, data in enumerate(train_dataloader):
        inputs, label = data
        inputs = inputs.cuda()
        label = label.cuda()
        outputs = net(inputs)
        one_hot_label = F.one_hot(label, num_classes=4)
        one_hot_label = one_hot_label.float()
        loss = F.mse_loss(outputs, one_hot_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch_idx % 10 == 0:
        #     print(epoch, batch_idx, loss.item())
    # print(epoch, loss.item())


for batch_idx, data in enumerate(train_dataloader):
    inputs, label = data
    inputs = inputs.cuda()
    label = label.cuda()
    outputs = net(inputs)

    # print(outputs)
    # print(label)
    break

with open('model_test.pt', 'wb') as f:
    torch.save(net.cpu().state_dict(), f)
