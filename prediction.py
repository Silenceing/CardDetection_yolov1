# -*- coding: utf-8 -*
from main import *


COLOR = [(255,0,0),(255,125,0),(255,255,0),(255,0,125),(255,0,250),
         (255,125,125),(255,125,250),(125,125,0),(0,255,125),(255,0,0),
         (0,0,255),(125,0,255),(0,125,255),(0,255,255),(125,125,255),
         (0,255,0),(125,255,125),(255,255,255),(100,100,100),(0,0,0),
         (0,255,100),(125,255,125),(255,100,255),(100,200,100)]

def draw_bbox(img,bbox,img0):
    h,w=img.shape[0:2]
    print(h,w)
    n=bbox.size()[0]
    # print("bbox:",bbox)
    # cv2.imshow("xx",img)
    # cv2.waitKey()
    H,W=img0.shape[0:2]
    gh=H/h
    gw=W/w

    # pred_result = [{"t  bg": image_name, "label_name": 'I', "bbox": [735, 923, 35, 75], "confidence": 0.2},
    #                 {"image_name": image_name, "label_name": 'I', "bbox": [525, 535, 53, 54], "confidence": 0.3}]

    box=[]
    # box=np.zeros(6)

    for i in range(n):
        p1=(int(w*bbox[i,1]*gw),int(h*bbox[i,2]*gh))
        p2 = (int(w * bbox[i, 3]*gw), int(h * bbox[i, 4]*gh))

        cls_name=classes[int(bbox[i,0])]
        confidence=bbox[i,5]


        print("bbox:",bbox)
        print(p1,p2,classes[int(bbox[i,0])],confidence.data.item())
        # img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img0)
        cv2.rectangle(img,p1,p2,color=COLOR[int(bbox[i,0])],thickness=2)
        cv2.putText(img,cls_name,p1,cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),thickness=2)

        box.append(cls_name)
        list_=[]
        list_.append(int(w * bbox[i, 1] * gw))
        list_.append(int(h * bbox[i, 2] * gh))
        list_.append(int(w * bbox[i, 3]*gw))
        list_.append( int(h * bbox[i, 4]*gh))
        box.append(list_)
        box.append(confidence.data.item())

    cv2.imwrite('./data/output/result/test.jpg',img0)
    cv2.imshow("box",img0)
    cv2.waitKey(0)
    return box

def xywh2xxyy(matrix):
    if matrix.size()[0:2] != (7, 7):
        raise ValueError("Error: Wrong labels size:", matrix.size())
    bbox = torch.zeros((98, 29))
    # 先把7*7*30的数据转变为bbox的(98,25)的格式，其中，bbox信息格式从(px,py,w,h)转换为(x1,y1,x2,y2),方便计算iou
    for i in range(7):
        for j in range(7):
            #box1的中心坐标
            bbox[2 * (i * 7 + j), 0:4] = torch.Tensor([(matrix[i, j, 0] + j) / 7 - matrix[i, j, 2] / 2,
                                                       (matrix[i, j, 1] + i) / 7 - matrix[i, j, 3] / 2,
                                                       (matrix[i, j, 0] + j) / 7 + matrix[i, j, 2] / 2,
                                                       (matrix[i, j, 1] + i) / 7 + matrix[i, j, 3] / 2])
            bbox[2 * (i * 7 + j), 4] = matrix[i, j, 4]
            bbox[2 * (i * 7 + j), 5:] = matrix[i, j, 10:]
            bbox[2 * (i * 7 + j) + 1, 0:4] = torch.Tensor([(matrix[i, j, 5] + j) / 7 - matrix[i, j, 7] / 2,
                                                           (matrix[i, j, 6] + i) / 7 - matrix[i, j, 8] / 2,
                                                           (matrix[i, j, 5] + j) / 7 + matrix[i, j, 7] / 2,
                                                           (matrix[i, j, 6] + i) / 7 + matrix[i, j, 8] / 2])
            bbox[2 * (i * 7 + j) + 1, 4] = matrix[i, j, 9]
            bbox[2 * (i * 7 + j) + 1, 5:] = matrix[i, j, 10:]

    return NMS(bbox)





def labels2bbox(matrix):
    """
        将网络输出的7*7*30的数据转换为bbox的(98,25)的格式，然后再将NMS处理后的结果返回
        :param matrix: 注意，输入的数据中，bbox坐标的格式是(px,py,w,h)，需要转换为(x1,y1,x2,y2)的格式再输入NMS
        :return: 返回NMS处理后的结果
        """
    if matrix.size()[0:2]!=(7,7):
        raise  ValueError("Error: Wrong labels size:",matrix.size())
    bbox=torch.zeros((98,29))
    # 先把7*7*30的数据转变为bbox的(98,25)的格式，其中，bbox信息格式从(px,py,w,h)转换为(x1,y1,x2,y2),方便计算iou
    for i in range(7):
        for j in range(7):
            bbox[2*(i*7+j),0:4]=torch.Tensor([(matrix[i,j,0]+j)/7-matrix[i,j,2]/2,
                                              (matrix[i,j,1]+i)/7-matrix[i,j,3]/2,
                                             (matrix[i,j,0]+j)/7+matrix[i,j,2]/2,
                                             (matrix[i,j,1]+i)/7+matrix[i,j,3]/2])
            bbox[2*(i*7+j),4]=matrix[i,j,4]
            bbox[2 * (i * 7 + j), 5:] = matrix[i, j, 10:]
            bbox[2 * (i * 7 + j)+1, 0:4] = torch.Tensor([(matrix[i, j, 5] + j) / 7 - matrix[i, j, 7] / 2,
                                                       (matrix[i, j, 6] + i) / 7 - matrix[i, j, 8] / 2,
                                                       (matrix[i, j, 5] + j) / 7 + matrix[i, j, 7] / 2,
                                                       (matrix[i, j, 6] + i) / 7 + matrix[i, j, 8] / 2])
            bbox[2 * (i * 7 + j)+1, 4] = matrix[i, j, 9]
            bbox[2 * (i * 7 + j)+1, 5:] = matrix[i, j,10:]
    print("------------ ",bbox)
    return NMS(bbox)


def NMS(bbox,conf_thresh=0.1,iou_thresh=0.3):
    """bbox数据格式是(n,29),前4个是(x1,y1,x2,y2)的坐标信息，第5个是置信度，后20个是类别概率
       :param conf_thresh: cls-specific confidence score的阈值
       :param iou_thresh: NMS算法中iou的阈值
       """
    n=bbox.size()[0]
    print("---n:",n)
    #bbox数据格式是(98,29)
    bbox_prob=bbox[:,5:].clone()#98行第5列之后均是概率(98,24)
    print("bbox_prob:",bbox_prob,bbox_prob.shape)
    #                  行           列
    bbox_conf1=bbox[:,4].clone().unsqueeze(1).expand_as(bbox_prob)#98行第4列置信度:将展开成98行24列，每行中列相等
    print("bbox_conf1:", bbox_conf1,bbox_conf1.shape)
    bbox_cls_spec_conf=bbox_conf1*bbox_prob#置信度*概率
    bbox_cls_spec_conf[bbox_cls_spec_conf<=conf_thresh]=0

    for c in range(24):
        rank=torch.sort(bbox_cls_spec_conf[:,c],descending=True).indices
        for i in range(n):#98
            if bbox_cls_spec_conf[rank[i],c]!=0:
                for j in range(i+1,98):
                    if bbox_cls_spec_conf[rank[j],c]!=0:
                        iou=calculate_iou(bbox[rank[i],0:4],bbox[rank[j],0:4])
                        if iou>iou_thresh:# 根据iou进行非极大值抑制抑制
                            bbox_cls_spec_conf[rank[j],c]=0

    bbox=bbox[torch.max(bbox_cls_spec_conf,dim=1).values>0]
    bbox_cls_spec_conf=bbox_cls_spec_conf[torch.max(bbox_cls_spec_conf,dim=1).values>0]
    res=torch.ones(bbox.size()[0],6)
    res[:,1:5]=bbox[:,0:4]
    res[:,0]=torch.argmax(bbox[:,5:],dim=1).int()
    res[:,5]=torch.max(bbox_cls_spec_conf,dim=1).values
    return res



class Prediction(FlyAI):
    def load_model(self):
        '''
        模型初始化，必须在此方法中加载模型
        '''
        # model=main.Net()
        model=torch.load("./data/output/model/epoch10.pkl")
        return model


    def predict(self, image_path):
        '''
        模型预测返回结果
        :参数示例 image_path='./data/input/image/0.jpg'
        :return: 返回预测结果格式具体说明如下：
        '''
        # val_data = DataLoader(main.LoadDataset(), batch_size=main.BatchSize, shuffle=True)
        image_path='./data/input/CardDetection/images/0.jpg'
        val_data = DataLoader(LoadDataset('./data/input/CardDetection/images/0.jpg'), batch_size=BatchSize, shuffle=True)
        image_name = os.path.basename(image_path) # 0.jpg
        #
        # # ... 模型预测
        #
        # # 返回bbox格式为 [xmin, ymin, width, height]
        # pred_result = [{"t  bg": image_name, "label_name": 'I', "bbox": [735, 923, 35, 75], "confidence": 0.2},
        #                 {"image_name": image_name, "label_name": 'I', "bbox": [525, 535, 53, 54], "confidence": 0.3}]
        #
        img0=cv2.imread('./data/input/CardDetection/images/0.jpg')

        model=self.load_model()

        dict={}
        pred_result=[]

        for i, (x, lb) in enumerate(val_data):
            print("x.shape",x.shape)
            pred = model(x)
            pred = pred.squeeze(dim=0)
            pred = pred.permute((1, 2, 0))
            # 此处可以用labels代替pred，测试一下输出的bbox是否和标签一样，从而检查labels2bbox函数是否正确。
            # 当然，还要注意将数据集改成训练集而不是测试集，因为测试集没有labels。
            bbox = labels2bbox(pred)#7*7*34
            x = x.squeeze(dim=0)  # 输入图像的尺寸是(1,3,448,448),压缩为(3,448,448)
            x = x.permute((1, 2, 0))  # 转换为(448,448,3)
            img = x.cpu().numpy()
            img = 255 * img  # 将图像的数值从(0,1)映射到(0,255)并转为非负整形
            img = img.astype(np.uint8)
            print(bbox)
            box=draw_bbox(img, bbox,img0)  # 将网络预测结果进行可视化，将bbox画在原图中，可以很直观的观察结果

            for i in range(len(box)//3):
                dict['image_name']=image_name
                dict['label_name']=box[i*3]
                dict['bbox'] = box[i * 3+1]
                dict['confidence'] = box[i * 3 + 2]
                pred_result.append(dict)
        print(pred_result)
        # return pred_result

if __name__=='__main__':
    prd=Prediction()
    prd.predict("")