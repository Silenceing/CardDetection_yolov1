# -*- coding: utf-8 -*-
import argparse
import os

from flyai.data_helper import DataHelper
from flyai.framework import FlyAI

from path import MODEL_PATH
from path import DATA_PATH
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transform
import torch
import torch.nn as nn
import torchvision.models as tvmodel
'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()

classes=['I' ,'II', 'III', 'IV','V','VI',
         'VII','VIII', 'IX','X','XI','XII',
         'XIII','XIV','XV','XVI', 'XVII','XVIII',
         'XIX','XX','XXI','XXII','XXIII','XXIV']

classNum=24
box_NUM=2
BatchSize=1
Epoch=100
Lr=0.0001
def bbox2labels(bbox):
    grid_size=1.0/7
    labels=np.zeros((7,7,5*box_NUM+classNum))

    for i in range(len(bbox)//5):
        gridx=int(bbox[i*5+1]//grid_size)
        gridy=int(bbox[i*5+2]//grid_size)

        gridpx=bbox[i*5+1]//grid_size-gridx
        gridpy=bbox[i*5+2]//grid_size-gridy

        #box+con设置
        labels[gridx,gridy,0:5]=np.array([gridpx,gridpy,bbox[i*5+3],bbox[i*5+4],1])
        labels[gridx,gridy,5:10]=np.array([gridpx,gridpy,bbox[i*5+3],bbox[i*5+4],1])
        #cls设置
        labels[gridx,gridy,10+int(bbox[i*5])]=1
    # print(labels.shape)
    return labels

class LoadDataset(Dataset):
    def __init__(self,datapath):
        self.filenames = []
        # trainpath = DATA_PATH + '/CardDetection/images/'
        self.imgpath = 'data/input/CardDetection/images/'
        self.labelpath = 'data/input/CardDetection/labels/'

        print(datapath)
        if os.path.isfile(datapath):
            file=datapath.split('/',5)[5]
            self.filenames.append(file.split('.')[0])
        else:
            for file in os.listdir(datapath):
                self.filenames.append(file.split('.')[0])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):

        img=cv2.imread(self.imgpath+self.filenames[item]+'.jpg')
        # img=cv2.imread('data/input/CardDetection/images/0.jpg')
        # print(img.shape)#h,w,channel
        h,w=img.shape[:2]
        input_size=448
        #图像增广
        padwh=(max(w,h)-min(w,h))//2
        if h>w:
            img=np.pad(img,((0,0),(padwh,padwh),(0,0)),'constant',constant_values=0)
        elif h<w:
            img = np.pad(img, ((padwh, padwh),(0, 0) , (0, 0)), 'constant', constant_values=0)
        #调整图像
        img=cv2.resize(img,(input_size,input_size))
        #
        # # cv2.imshow('p',img)
        img=torch.from_numpy(img.transpose(2, 0, 1)).float()

        # cv2.waitKey()
        # print(img)


        with open(self.labelpath+self.filenames[item]+".txt") as f:
        # with open('data/input/CardDetection/labels/0.txt')as f:
            bbox=f.read().split('\n')

        bbox=[x.split() for x in bbox]
        # 字符转数字,
        bbox=[float(x) for y in bbox for x in y]
        # print(bbox)

        #每个text文件中数据：标签、x,y,w,h
        for i in range(len(bbox)//5):
        #     x1, y1 = (int(bbox[i*5+1] * w - bbox[i*5+3] * w / 2), int(bbox[i*5+2] * h - bbox[i*5+4] * h / 2))  # 左上
        #     x2, y2 = (int(bbox[i*5+1] * w + bbox[i*5+3] * w / 2), int(bbox[i*5+2] * h + bbox[i*5+4] * h / 2))  # 右下
        #     print(x1, y1, x2, y2)
        #     # ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
        #     cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 255, 0), thickness=3)
        #     cv2.putText(img, classes[int(bbox[i*5])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=3)
        # cv2.imwrite('./testpic/' + "origin" + ".jpg", img)
        # cv2.imshow('pic', img)
        # cv2.waitKey()

            #增广后(max(w,h),max(w,h))微调x,y,w,h
            if h>w:
                bbox[i * 5 + 1] = (bbox[i * 5 + 1] * w+ padwh) / h
                bbox[i * 5 + 3] = (bbox[i * 5 + 3] * w) / h
            elif w>h:
                bbox[i * 5 + 2] = (bbox[i * 5 + 2] * h+ padwh) / w
                bbox[i * 5 + 4] = (bbox[i * 5 + 4] * h) / w
        # #
        #     x1, y1 = (int(bbox[i*5+1] * max(w,h) - bbox[i*5+3] * max(w,h) / 2), int(bbox[i*5+2] * max(w,h) - bbox[i*5+4] * max(w,h) / 2))  # 左上
        #     x2, y2 = (int(bbox[i*5+1] * max(w,h) + bbox[i*5+3] * max(w,h) / 2), int(bbox[i*5+2] * max(w,h) + bbox[i*5+4] * max(w,h) / 2))  # 右下
        #     print(x1, y1, x2, y2)
        #     # ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
        #     cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 255, 0), thickness=2)
        #     cv2.putText(img, classes[int(bbox[i*5])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
        # cv2.imwrite('./testpic/' + "pading" + ".jpg", img)
        # cv2.imshow('pic', img)
        # cv2.waitKey()


        #     x1, y1 = (int(bbox[i*5+1] * input_size - bbox[i*5+3] * input_size / 2), int(bbox[i*5+2] * input_size - bbox[i*5+4] * input_size / 2))  # 左上
        #     x2, y2 = (int(bbox[i*5+1] * input_size + bbox[i*5+3] * input_size / 2), int(bbox[i*5+2] * input_size + bbox[i*5+4] * input_size / 2))  # 右下
        #     print(x1, y1, x2, y2)
        #     # ascontiguousarray函数将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快
        #     cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 255, 0), thickness=1)
        #     cv2.putText(img, classes[int(bbox[i*5])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=1)
        #
        #
        # cv2.imwrite('./testpic/' + 'resize' + ".jpg", img)
        # cv2.imshow('pic', img)
        # cv2.waitKey()
        #
        labels=bbox2labels(bbox)
        labels=transform.ToTensor()(labels)

        # print("labels.shape:", labels.shape)
        # print("img.shape:", img.shape)

        return img,labels




        # print(filenames)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        resnet=tvmodel.resnet34(pretrained=True)
        resnet_out_channel=resnet.fc.in_features
        self.resnet=nn.Sequential(*list(resnet.children()))[:-2]

        self.convlayer=nn.Sequential(
            nn.Conv2d(resnet_out_channel,1024,3,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.Conv2d(1024,1024,3,stride=2,padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU()
        )
        self.fclayer=nn.Sequential(
            nn.Linear(7*7*1024,4096),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 7 * 7 * 34),
            nn.Sigmoid()
        )
    def forward(self,out):
        out=self.resnet(out)
        out=self.convlayer(out)
        out=out.view(out.size()[0],-1)
        out=self.fclayer(out)

        return out.reshape(-1,(5*box_NUM+classNum),7,7)

def calculate_iou(bbox1,bbox2):
    intersect_bbox=[0.,0.,0.,0.]
    #真实框与预测框不重叠的4种情况
    if bbox1[2]<bbox2[0] or bbox1[0]>bbox2[2]or bbox1[3]<bbox2[1]or bbox1[1]>bbox2[3]:
        pass
    else:
        #重叠区域坐标
        intersect_bbox[0]=max(bbox1[0],bbox2[0])
        intersect_bbox[1] = max(bbox1[1], bbox2[1])
        intersect_bbox[2] = min(bbox1[2], bbox2[2])
        intersect_bbox[3] = min(bbox1[3], bbox2[3])

    area1=(bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    #重叠区域面积
    area_intersect=(intersect_bbox[2]-intersect_bbox[0])*(intersect_bbox[3]-intersect_bbox[1])

    if area_intersect>0:
        return area_intersect/(area1+area2-area_intersect)
    else:
        return 0

class objloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,pred,labels):

        numgridx,numgridy=labels.size()[-2:]
        noobj_confi_loss=0. # 不含目标的网格损失(只有置信度损失)
        coor_loss=0.#含有目标的bbox的坐标损失
        n_batch=labels.size()[0]
        class_loss=0.# 含有目标的网格的类别损失
        obj_conf_loss=0. # 含有目标的bbox的置信度损失

        for i in range(n_batch):
            for n in range(7):
                for m in range(7):
                    if labels[i,4,m,n]==1:# 如果包含物体
                        #pred:batch,[x1,y1,w1,h1,conf1,x2,y2,w2,h2,conf2,cls1,cls2....cls20],m,n
                        bbox1_pred_xyxy=((pred[i,0,m,n]+m)/numgridx-pred[i,2,m,n]/2,
                                         (pred[i,1,m,n]+n)/numgridy-pred[i,3,m,n]/2,
                                         (pred[i,0,m,n]+m)/numgridx+pred[i,2,m,n]/2,
                                         (pred[i,1,m,n]+n)/numgridy+pred[i,3,m,n]/2)

                        bbox2_pred_xyxy = ((pred[i, 5, m, n] + m) / numgridx - pred[i, 7, m, n] / 2,
                                           (pred[i, 6, m, n] + n) / numgridy - pred[i, 8, m, n] / 2,
                                           (pred[i, 5, m, n] + m) / numgridx + pred[i, 7, m, n] / 2,
                                           (pred[i, 6, m, n] + n) / numgridy + pred[i, 8, m, n] / 2)

                        bbox_gt_xxyy= ((labels[i, 0, m, n] + m) / numgridx - labels[i, 2, m, n] / 2,
                                           (labels[i, 1, m, n] + n) / numgridy - labels[i, 3, m, n] / 2,
                                           (labels[i, 0, m, n] + m) / numgridx + labels[i, 2, m, n] / 2,
                                           (labels[i, 1, m, n] + n) / numgridy + labels[i, 3, m, n] / 2)

                        iou1=calculate_iou(bbox1_pred_xyxy,bbox_gt_xxyy)
                        iou2=calculate_iou(bbox2_pred_xyxy,bbox_gt_xxyy)

                        if iou1>=iou2:# 选择iou大的bbox作为负责物体
                            coor_loss=coor_loss+5*(torch.sum((pred[i,0:2,m,n]-labels[i,0:2,m,n])**2)+torch.sum((pred[i,2:4,m,n].sqrt()-labels[i,2:4,m,n].sqrt())**2))

                            obj_conf_loss=noobj_confi_loss+(pred[i,4,m,n]-iou1)**2
                            # iou比较小的bbox不负责预测物体，因此confidence loss算在noobj中，注意，对于标签的置信度应该
                            noobj_confi_loss=noobj_confi_loss+0.5*((pred[i,9,m,n]-iou2)**2)
                        else:
                            coor_loss=coor_loss+5*(torch.sum((pred[i,5:7,m,n]-labels[i,5:7,m,n])**2)+torch.sum((pred[i,7:9,m,n].sqrt()-labels[i,7:9,m,n].sqrt())**2))
                            obj_conf_loss=obj_conf_loss+(pred[i,9,m,n]-iou2)**2
                            noobj_confi_loss=noobj_confi_loss+0.5*((pred[i,4,m,n]-iou1)**2)
                        class_loss=class_loss+torch.sum((pred[i,10:,m,n]-labels[i,10:,m,n])**2)
                    else:
                        noobj_confi_loss+=0.5*torch.sum(pred[i,[4,9],m,n]**2)
        loss=coor_loss+obj_conf_loss+noobj_confi_loss+coor_loss
        return loss/n_batch


class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''

    def download_data(self):
        # 根据数据ID下载训练数据
        data_helper = DataHelper()
        data_helper.download_from_ids("CardDetection")

    def deal_with_data(self):
        '''
        处理数据，没有可不写。
        :return:
        '''
        # pass
        if not os.path.exists(DATA_PATH+'/CardDetection/labels'):
            os.makedirs(DATA_PATH+'/CardDetection/labels')
        datapath=DATA_PATH+'/CardDetection/train.csv'
        data=pd.read_csv(datapath)

        # print()
        for j in range(len(data)):
            for i,cls in enumerate(classes):
                if data.iloc[j,1]==cls:
                    data.iloc[j,1]=i
                    # print(data.iloc[j]['label'])
        label = data['label']
        img = data['image_path']
        # print(data)

        trainpath=DATA_PATH+'/CardDetection/images/'
        train_file=[]
        for file  in os.listdir(trainpath):
            # file=file.strip('.')[0]
            # print('images/'+file)
            train_file.append('images/'+file)
        # print(label.values)


        for file in train_file:
            sp = file.split('/')[1]
            sp = sp.split('.')[0]
            out_file = open('./data/input/CardDetection/labels/%s.txt' % (sp), 'w')
            img = cv2.imread(DATA_PATH + '/CardDetection/' + file)
            h, w = img.shape[:2]
            dw = 1.0 / w
            dh = 1.0 / h
            for i in range(len(data)):
                if file==data.iloc[i,0]:
                    # img=cv2.imread(DATA_PATH+'/CardDetection/'+file)
                    # cv2.imshow('test',img)
                    # cv2.waitKey()
                    # print(data.iloc[i,0])
                    xc=(data.iloc[i,4]+data.iloc[i,2])/2.0
                    yc=(data.iloc[i,5]+data.iloc[i,3])/2.0
                    new_w=data.iloc[i,4]-data.iloc[i,2]
                    new_h=data.iloc[i,5]-data.iloc[i,3]

                    xc*=dw
                    yc*=dh
                    new_w*=dw
                    new_h*=dh
                    out_file.write(str(data.iloc[i,1])+" "+str(xc)+" "+str(yc)+" "+str(new_w)+" "+str(new_h)+'\n')

        # print(img.values)
        # print(data)



    def train(self):
        '''
        训练模型，必须实现此方法
        :return:
        '''
        modelpath='./data/output/model/'
        data = LoadDataset(DATA_PATH + '/CardDetection/images/')
        train_ld=DataLoader(LoadDataset(DATA_PATH + '/CardDetection/images/'),batch_size=BatchSize,shuffle=True)
        # device = torch.device('cuda:0')
        # model=Net().to(device)

        model=Net()


        for ly in model.children():
            ly.required_grad=False
            break

        loss=objloss()
        optim=torch.optim.SGD(model.parameters(),lr=Lr,momentum=0.9,weight_decay=0.0005)

        for e in range(Epoch):
            model.train()
            for i,(x,y) in enumerate(train_ld):
                # x,y=x.float().cuda(),y.float().cuda()
                x, y = x.float(), y.float()
                ypred=model(x)
                loss_=loss(ypred,y)
                optim.zero_grad()
                loss_.backward()
                optim.step()
                print('Epoch:[%d/%d]  step: [%d/%d]  loss:%.2f'%(e+1,Epoch,i,len(data)//BatchSize,loss_))
            if (e+1)%10==0:
                torch.save(model,modelpath+'epoch'+str(e+1)+'.pkl')




if __name__ == '__main__':
    main = Main()
    # main.download_data()

    main.train()
    # main.deal_with_data()
    # LoadDataset()
    # print(img,lb)
    # x=torch.randn(1,3,448,448)
    # net=Net()
    # print(net)
    # y=net(x)
    # print(y.size())
