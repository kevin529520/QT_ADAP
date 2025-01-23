# data loader
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import OrderedDict
import torch.optim as optim
import numpy as np
import cv2
import random
import time

####################################################################################
######################### Date Loader ##############################################
####################################################################################

def loadTxt(label_file = "p1-6_predict.txt"):
    temp = np.loadtxt(label_file, dtype=str)
    image_name = []
    label_list = []
    for i in range(len(temp)):
        tempX = temp[i].split(',')
        # tempLabel = list(map(float, tempX[1:-2]))
        tempLabel = list(map(float, tempX[1:]))
        tempLabel = np.array(tempLabel)
        tempImageName = tempX[0]

        image_name.append(tempImageName)
        label_list.append(tempLabel)
    # image name list, label list [weldX, weldY,circleX, circleY, circleR]
    return image_name, np.array(label_list)

# 样本随机采样
def splitDate(sample_list, rate=0.8):  # 0.8
    index_list = list(range(len(sample_list)))
    train_num = int(len(sample_list)*rate)
    random.seed(42)
    train_sample_index = random.sample(index_list,train_num)
    var_sample_index = list(set(index_list)-set(train_sample_index))
    print('var_sample_index:', var_sample_index)
    return train_sample_index, var_sample_index


def randmonSampleData(label_filename):
    data_name,data_label = loadTxt(label_filename)
    # # change to dict
    # data_dict = {k: v for k, v in zip(data_name, data_label)}
    # split data
    train_index, var_index = splitDate(data_name, rate=0.8) # 0.8
    
    # train_label = [data_dict.get(key) for key in train_sample_img]
    # var_label = [data_dict.get(key) for key in var_sample_img]

    train_label = [data_label[i] for i in train_index]
    train_img = [data_name[i] for i in train_index]
    var_label = [data_label[i] for i in var_index]
    var_img = [data_name[i] for i in var_index]
    # [train image name list, train label list], [var image name list, var label list]
    return [train_img, train_label],[var_img, var_label]


####################################################################################
######################### Date Loader ##############################################
####################################################################################

class MyDataset(Dataset):
    def __init__(self, data_list, img_dir):
        self.img_name, self.img_labels = data_list[0], data_list[1]
        self.dir = img_dir
        self.scale = 0.5

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.dir, self.img_name[idx])

        img_data = cv2.imread(img_path)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        img_data = img_data/255.0
        w = int(img_data.shape[1]*self.scale)
        h = int(img_data.shape[0]*self.scale)
        img_data = cv2.resize(img_data, dsize=(w, h))

        weld_pos = self.img_labels[idx][0:2]*self.scale
        label = self.img_labels[idx][2:]*self.scale

        return torch.Tensor(np.array([img_data])), torch.from_numpy(weld_pos), torch.from_numpy(label)
    
#################################################################################################
################################# ResNet ########################################################
#################################################################################################
def conv3x3(in_channel, out_channel, stride=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    kernel_size = np.asarray((3,3))
    # Compute the size of the upsampled filter with
    # a specified dilation rate.
    upsampled_kernel_size = (kernel_size-1)*(dilation-1) + kernel_size
    # Determine the padding that is necessary for full padding,
    # meaning the output spatial size is equal to input spatial size
    full_padding = (upsampled_kernel_size-1)//2
    # Conv2d doesn't accept numpy arrays as arguments
    full_padding, kernel_size = tuple(full_padding), tuple(kernel_size)

    return nn.Conv2d(
        in_channel,
        out_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=full_padding,
        dilation=dilation,
        bias=bias,
    )


class ResBlock(nn.Module):
    """ basic resNet unit"""
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(ResBlock, self).__init__()

        self.stride = stride
        self.dilation = dilation
        self.downsample = downsample

        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes*self.expansion, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes*self.expansion)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
    
class ResNet(nn.Module):
    """    A fully-convolutional network with an encoder-decoder architecture.
    """
    hid_dims = [32, 64,128,128]

    def __init__(self):
        super().__init__()

        self.featureExtractor = nn.Sequential(
            OrderedDict(
                [
                    (
                        "enc_conv0",
                        nn.Conv2d(
                            1,
                            self.hid_dims[0],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    ("enc_norm0", nn.BatchNorm2d(self.hid_dims[0])),
                    ("enc_relu0", nn.ReLU(inplace=True)),
                    ("enc_pool1", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                    (
                        "enc_resn2",
                        ResBlock(
                            self.hid_dims[0],
                            self.hid_dims[1],
                            downsample=nn.Sequential(
                                nn.Conv2d(self.hid_dims[0], self.hid_dims[1], kernel_size=1, bias=True),
                                nn.BatchNorm2d(self.hid_dims[1]),
                            ),
                            dilation=1,
                        ),
                    ),
                    ("enc_pool3", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                    (
                        "enc_resn4",
                        ResBlock(
                            self.hid_dims[1],
                            self.hid_dims[2],
                            downsample=nn.Sequential(
                                nn.Conv2d(self.hid_dims[1], self.hid_dims[2], kernel_size=1, bias=True),
                                nn.BatchNorm2d(self.hid_dims[2]),
                            ),
                            dilation=1,
                        ),
                    ),
                    # (
                    #     "enc_resn5",
                    #     ResBlock(
                    #         self.hid_dims[2],
                    #         self.hid_dims[3],
                    #         downsample=nn.Sequential(
                    #             nn.Conv2d(self.hid_dims[2], self.hid_dims[3], kernel_size=1, bias=True),
                    #             nn.BatchNorm2d(self.hid_dims[3]),
                    #         ),
                    #         dilation=1,
                    #     ),
                    # ),
                    (
                        "enc_conv1",
                        nn.Conv2d(
                            self.hid_dims[2],
                            1,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True,
                        ),
                    ),
                    ("enc_norm1", nn.BatchNorm2d(1),),
                    ("enc_pool5", nn.AdaptiveMaxPool2d((30, 30)))
                ]
            )
        )

        self.classifier = nn.Sequential(
            nn.Linear(30 * 30 +2, 500),#128
            nn.ReLU(inplace=True),
            nn.Linear(500, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 100),
            nn.Linear(100, 3),
        )

        # self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # if m.bias is not None:
                    # nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, weld_pos):
        out_features = self.featureExtractor(x)
        out_features = torch.flatten(out_features, start_dim=1)

        # vector stack
        fusion_feature = torch.cat((out_features,weld_pos),axis = 1)
        result = self.classifier(fusion_feature)
        
        return result



####################################################################################
######################### train, var ####################################################
#####################################################################################
class ContourPredictor(object):
    def __init__(self, data_folder,label_file,saved_pth_path):
        self.batch_size = 32
        self.learning_rate = 0.0005
        self.momentum = 0.3
        self.epoch = 200

        # self.saved_pth = "test_weight.pth"
        # self.data_folder = "p1-6_cropped_renamed"
        # self.label_filename = "p1-6_predict.txt"
        self.saved_pth = saved_pth_path
        self.data_folder = data_folder
        self.label_filename = label_file
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        # self.device = torch.device("cpu")

        self.train_dataset, self.var_dataset= randmonSampleData(self.label_filename)
        # [train_sample_img, train_label],[var_sample_img, var_label] 


    def train(self):
        print("sdsd:",self.label_filename)
        # load data
        dataset_image = MyDataset(data_list=self.train_dataset, img_dir=self.data_folder)
        print("sample numbers: ", len(dataset_image))
        traing_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=True, num_workers=2, drop_last=False, pin_memory=False, collate_fn=None)
        
        # model = AlexNet().to(self.device)
        model = ResNet().to(self.device)

        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        vloss = torch.nn.MSELoss()

        for epoch in range(self.epoch):
            # training
            model.train()
            running_loss = 0.0
            cnt = 0
            for i, datax in enumerate(traing_data):
                # model.train()
                temp_img= datax[0]
                temp_weld_pos= datax[1]
                temp_label = datax[2]

                temp_weld_pos = temp_weld_pos.to(torch.float32)
                temp_label = temp_label.to(torch.float32)

                temp_img= temp_img.to(self.device)
                temp_weld_pos= temp_weld_pos.to(self.device)
                temp_label = temp_label.to(self.device)

                optimizer.zero_grad()
                outputs = model(temp_img, temp_weld_pos)
                print('outputs:', outputs)
                
                # loss = vloss(outputs, temp_label)
                a,b = self.circleLossSumTensor(outputs, temp_label)
                loss = a +b 
                running_loss += loss.item()

                # print("epoch ", epoch, " i ",i, loss.item())
                
                loss.backward()
                optimizer.step()

                cnt += 1

            # if epoch % 4 == 3:
            #     print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss/4))
            #     running_loss = 0.0
            # print('[epoch %d] loss: %.4f' % (epoch + 1, running_loss/(len(dataset_image)/self.batch_size)))
            loss_record.append(running_loss/cnt)
            print('[epoch %d] loss: %.4f' % (epoch + 1, running_loss/cnt))

            # save weight
        torch.save(model.state_dict(), self.saved_pth)


    @torch.no_grad()
    def var(self, pre_pth="test_weight.pth"):
        # load data
        dataset_image = MyDataset(data_list=self.var_dataset, img_dir=self.data_folder)
        print("sample numbers: ", len(dataset_image))
        traing_data = torch.utils.data.DataLoader(dataset=dataset_image, batch_size=self.batch_size, shuffle=False, num_workers=2, drop_last=False, pin_memory=False, collate_fn=None)
        
        model = ResNet().to(self.device)
        model.load_state_dict(torch.load(pre_pth))
        # loss
        vloss = torch.nn.MSELoss()

        # var
        model.eval()
        result_a = 0
        result_b = 0
        cnt = 0
        prediction_times = []
        for i, datax in enumerate(traing_data):
            temp_img= datax[0]
            temp_weld_pos= datax[1]
            temp_label = datax[2]
            print('temp_label:', temp_label)

            temp_weld_pos = temp_weld_pos.to(torch.float32)
            temp_label = temp_label.to(torch.float32)

            temp_img= temp_img.to(self.device)
            temp_weld_pos= temp_weld_pos.to(self.device)
            temp_label = temp_label.to(self.device)

            start_time = time.time()
            outputs = model(temp_img, temp_weld_pos)
            end_time = time.time()
            prediction_time = end_time - start_time
            prediction_times.append(prediction_time)

            print('outputs:',outputs)
            loss = vloss(outputs, temp_label)
            # print(i, loss, outputs, temp_label)
            a,b = self.circleLossSum(outputs, temp_label)
            cnt = cnt + len(outputs)
            result_a = result_a + a
            result_b = result_b + b
            print(i,a/len(outputs),b/len(outputs))
            print(i, loss)
        print("center error, radius error:",result_a/cnt, result_b/cnt)
        average_prediction_time =  sum(prediction_times) / 240
        print('average_prediction_time:', average_prediction_time)

    def circleLossSum(self, predicted, target):
        # compute radius and center error

        predicted = predicted.cpu().detach().numpy()
        target = target.cpu().detach().numpy()
        if len(predicted) != len(target):
            raise TypeError("the shape of output and target is not equal!")
        batch_size = len(predicted)
        centrt_err= 0
        radius_err = 0
        for i in range(batch_size):
            centrt_err = centrt_err + np.linalg.norm(predicted[i][0:2] - target[i][0:2])
            radius_err = radius_err + abs(predicted[i][2]-target[i][2])
        
        return centrt_err, radius_err
    
    def circleLossSumTensor(self, predicted, target):
        # compute radius and center error

        # predicted = predicted.cpu().detach().numpy()
        # target = target.cpu().detach().numpy()
        if len(predicted) != len(target):
            raise TypeError("the shape of output and target is not equal!")
        batch_size = len(predicted)
        centrt_err= 0
        radius_err = 0
        for i in range(batch_size):
            centrt_err = centrt_err + torch.norm(predicted[i][0:2] - target[i][0:2])
            radius_err = radius_err + torch.abs(predicted[i][2]-target[i][2])
        
        return centrt_err/batch_size, radius_err/batch_size




if __name__ == "__main__":
    # test_dataset = CustomImageDataset(annotations_file='no_position_predict.txt', img_dir='dataset_no_position')
    # data_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=1, drop_last=False)
    # print('============')
    # for batch_id, (images, pos, labels) in enumerate(data_loader):
    #     print(batch_id, images.shape,pos.shape, labels.shape)

    data_folder = "./Data_mini/240301_mini"
    label_filename = "./Data_mini/labels-0301_mini.txt"
    saved_pth_name = 'weight_0301_mini.pth'

    # data_folder = "./Data_mini_var/240301_mini_var"
    # label_filename = "./Data_mini_var/labels-0301_mini_var.txt"
    # saved_pth_name = 'weight_0301_mini.pth'

    # data_folder = "./Data_var/240301_var"
    # label_filename = "./Data_var/labels-0301_var.txt"
    # saved_pth_name = 'weight_0301.pth'
    # average_prediction_time = 0
    loss_record = []

    a = ContourPredictor(data_folder=data_folder,label_file=label_filename,saved_pth_path=saved_pth_name)

    a.train()
    # a.var(saved_pth_name)
  
    # # 将损失记录存储到txt文件
    # with open('loss_record.txt', 'w') as file:
    #     for loss in loss_record:
    #         file.write(str(loss) + '\n')

    # # 绘制损失值图表
    # plt.plot(loss_record, label='Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training Loss over Epochs')
    # plt.legend()
    # plt.show()
