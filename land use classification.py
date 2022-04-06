import torch
import os
import pickle
import pandas as pd
import json
import logging
import time
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torch import nn
import numpy as np
#import tensorwatch as tw
from torch.nn import functional as F
import matplotlib.font_manager as fm 
from PIL import Image,ImageTk,ImageDraw,ImageFont
import math

class config:
    PATH_GT_TRAIN="/Public/TianYu/NewTest/Datasets/BEAUTY1.0_COCOLike_8_16/Split0/annotations/train_reb.json"
    PATH_GT_TEST="/Public/TianYu/NewTest/Datasets/BEAUTY1.0_COCOLike_8_16/Split0/annotations/test.json"
    PATH_GT_VAL="/Public/TianYu/NewTest/Datasets/BEAUTY1.0_COCOLike_8_16/Split0/annotations/val.json"
    
    PATH_PR={'faster_rcnn_r50':'/Public/TianYu/NewTest/COCO/workdir_coco_9_2/faster_rcnn_r50_fpn_2x_coco/',
          'cascade_rcnn_r101':'/Public/TianYu/NewTest/workdir_coco_5_18/scnet_r101_fpn_20e_coco/',
          'faster_rcnn_r101':'/Public/TianYu/NewTest/COCO/workdir_coco_9_2/faster_rcnn_r101_fpn_2x_coco/',
          'cascade_rcnn_r50':'/Public/TianYu/NewTest/workdir_coco_5_18/scnet_r50_fpn_20e_coco/'}
    
  
    
    labels=['Commercial','Residential','Industrial','Public']
    bboxs=['retail','house','roof','officebuilding','apartment','garage','industrial','church']
    labels2idx={label:idx for idx,label in enumerate(labels)}
    bboxs2idx={bbox:idx for idx,bbox in enumerate(bboxs)}
    
    WORK_DIR='/Public/TianYu/NewTest/beauty_attention_work_dir_5_24'
    BATCH_SIZE=64
    #(8,128,4,1,False,'LSTM')
    #(self, input_size, hidden_size, output_size,num_layers,bidirectional,rnn_type):#输入，隐藏层单元，输出，隐藏层数，双向
    HIDDEN_SIZE=128
#     NUM_LAYERS=2
#     BIDIRECTIONAL='False'
#     RNN_TYPE='LSTM'#/GRU
    NUM_EPOCHS=10
    LR=0.01
    lr_period, lr_decay= 10, 0.5
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matrix=[[0.7, 0.01, 0.05, 0.07, 0.01, 0.02, 0.12, 0.02],
[0.01, 0.81, 0.01, 0.02, 0.03, 0.09, 0.01, 0.03],
[0.09, 0.01, 0.88, 0.01, 0.0, 0.0, 0.01, 0.0],
[0.09, 0.03, 0.0, 0.67, 0.01, 0.12, 0.06, 0.02],
[0.03, 0.1, 0.01, 0.02, 0.83, 0.0, 0.02, 0.0],
[0.02, 0.12, 0.0, 0.16, 0.0, 0.62, 0.02, 0.04],
[0.1, 0.0, 0.0, 0.04, 0.01, 0.01, 0.82, 0.03],
[0.04, 0.04, 0.0, 0.07, 0.0, 0.02, 0.04, 0.8]]
    #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    Skip_Gram_wordvector=[[0.05200207, -0.05823211, 0.07790990, 0.03911300],
[0.05762698, -0.03808329, -0.06816287, -0.04007124],
[0.04404941, 0.05943516, -0.00579134, -0.05371539],
[0.02371268, -0.11500797, -0.04955185, 0.08241102],
[0.09706229, 0.04375809, -0.00283467, -0.08233096],
[0.07199545, 0.04275675, -0.08876450, -0.00131866],
[-0.02889932, 0.00103070, 0.00291926, -0.08482169],
[0.06266880, 0.08602013, 0.02788937, -0.00192571]
 ]

    Skip_Gram2_02_wordvector = [[  3.0684, -10.3071,  -8.7243,  -7.8877,   2.3433,  12.4539,  10.6761, -5.6425],
        [  3.8138,  -6.0471,  -1.8226,  -3.0945,   1.0296,   0.7067,  11.1167, -9.8869],
        [ -0.3531,   7.4326,   5.1660,  -8.0937,   9.9487,   8.8652,   9.0649, -2.9154],
        [ -2.9722,   2.9427,  -2.6679,  -7.4401,   9.8853,  -2.6532,   0.2075, -9.1798],
        [  4.8983,  -5.0190,   1.2351,  -5.4163,  -0.4946,   1.5610,  -1.3789, -9.4346],
        [  8.2524,  -0.7044,  -1.9818,   9.4340, -12.4776,  -6.3602, -11.6418, -10.1261],
        [  0.6728,   1.4611,   2.9496,   4.5700,   0.3035,   2.2372,  -5.7210, -9.5451],
        [ -0.7961,  -5.5606,   7.7136,  -6.3232,  -1.6874,  -3.4914, -10.0549, -7.6057]]
        
    Skip_Gram3_02_wordvector=[[-3.9788,  0.0429,  1.2131, -1.6222,  4.8363,  4.7079, -3.6099, -3.5002,3.9604,  4.3734],
        [ 0.1405, -0.0724,  3.5769,  2.7894, -0.5601,  4.9269, -2.8006,  3.8679,-3.6956, -2.5450],
        [-4.9650, -3.6594, -2.5811, -0.9162,  1.9057, -1.5423,  3.2424,  3.1021,-0.7952,  2.7081],
        [-1.9916, -4.2004,  4.3145,  3.6619,  3.0516,  2.3274, -2.6175,  2.1818,-4.6577, -0.7700],
        [ 0.3781,  2.7194, -2.8533, -2.2509, -1.6221, -0.7328, -3.6957, -4.9843,0.9148, -2.6116],
        [-2.5245, -2.0602,  2.1817,  0.8263, -4.6393, -4.0271, -0.3733,  0.8248,4.0476,  3.7254],
        [-1.3638, -3.8269,  3.6770,  2.5957, -3.6528,  3.5542, -3.2934, -4.7150,2.3613,  4.2238],
        [ 3.7249,  3.2484,  2.8049,  4.6197, -2.4817,  4.2434,  3.1971, -3.7667,-1.2085, -3.6816]]
 
    
    CBOW_wordvector2=[[ 0.12341236, -0.01232588,  0.01762320, -0.06080350],
            [-0.07830755, 0.01611204, -0.00578560, -0.02674788],
             [ 0.00517364, -0.01349713, -0.01981429,  0.04426415],
            [-0.04588775, -0.04722403, -0.05106120, -0.02503637],
             [-0.02900910, -0.04796851, 0.01203516, -0.05104259],
            [ 0.03611496,  0.05536033,  0.04836035, -0.09754882],
           [ 0.10302308,  0.02738266,  0.03811255, -0.01884597],
           [-0.03057625, 0.05365050, 0.01354899, 0.02271561]
 ]    
    
    GloVe_wordvector3 = [
[0.093016, 0.096530, 0.013819, -0.123586],
[0.138321, -0.125809, 0.116470, 0.002157],
[0.013746, -0.126929, 0.065184, 0.120061],
[0.099826, -0.026109, -0.042779, -0.063206],
[-0.181200, -0.016065, -0.286654, 0.170516],
[0.120662, -0.027210, -0.283282, -0.001902],
[-0.090759, 0.169389, 0.213905, -0.055491],
[0.019704, -0.087109, -0.015703, -0.063019]
              ]
    GloVe_wordvector3_02 = [[-0.034010, 0.035612, -0.058817, 0.049280, 0.062090, 0.003474, -0.021807, -0.035120, 0.064124 ,-0.004013],
               [0.039963, -0.032135, 0.011509, -0.059963, -0.071785, -0.056637, 0.100521, 0.005227, -0.086316, 0.046245],
               [0.092985, 0.102349, -0.061135, 0.018775, 0.057510, -0.043719, -0.028717, 0.024083, 0.086005, 0.000930],
               [0.034041, 0.056259, 0.001911, -0.068159, 0.161500, -0.043550, -0.145746 ,-0.058563, 0.086552, -0.058238],
               [-0.006384, -0.129292, 0.136627, 0.075218, -0.179943, 0.070505, 0.086771, 0.054906, -0.030774, -0.049041],
               [-0.009067, 0.121706, 0.006922,0.002934 ,0.096710, 0.084213, -0.063445, -0.029858, -0.025054, -0.081733],
               [0.003780, -0.040385, -0.141834, -0.043095, 0.097040 ,-0.033756, -0.074865, 0.062869, 0.123836, 0.047137],
               [0.037983, 0.073207, 0.018063, -0.029170 ,0.008037, -0.041160, -0.054340, 0.010613, -0.027770, 0.026654]]
    
def get_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(fmt="%(message)s")

    sHandler = logging.StreamHandler()
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    if not os.path.exists(config.WORK_DIR+'/log'):
        os.makedirs(config.WORK_DIR+'/log')
        #LOG_DIR +'/'+ RNN_TYPE+'_'+BIDIRECTIONAL+'_split'+str(SPLIT)+'.log'
    localtime = time.asctime( time.localtime(time.time()) )
    fHandler = logging.FileHandler("{}/{}.log".format(config.WORK_DIR+'/log','5_18'), mode='w')
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)
    return logger

def get_pr_list(path_pkl,path_gt,confidence_thr=0.0,merge_iou=False,merge_iou2=False,bbox_num_thr=0):
    #merge_iou，同一位置只保留score最大的,merge_iou2：相同位置的归类，但是保留
    #bbox_num_thr等于2，则把一个bbox和两个bbox的image全pass了。以此类推
    with open(path_pkl,'rb') as f:
        pkl_data = pickle.load(f)
    with open(path_gt) as f:
        json_data=json.load(f)
        
    #list一个元素是一张图片
    #一张图片有多个bbox列表
    #一个bbox列表元素有 图片名id  bboxid  bbox种类  图片名字
    #将pkl预测文件转化为list
    #reb变化有三种：_HorizontalFlip _hue 和 
    pr_list=[]
        
    for image_idx,image in enumerate(pkl_data):
        
        image_list=[]
        bbox_class='None'
        if('_hue' in json_data['images'][image_idx]['file_name']):
            
            image_name_t=json_data['images'][image_idx]['file_name'].split('_hue.jpg')[0]
        elif('_HorizontalFlip' in json_data['images'][image_idx]['file_name']):
            image_name_t=json_data['images'][image_idx]['file_name'].split('_HorizontalFlip.jpg')[0]
        else:
            image_name_t=json_data['images'][image_idx]['file_name'].split('.jpg')[0]
            #for class_name in class_data:#查找分类文件的四个种类dict
            #    if(image_name_t in class_data[class_name]):
            #        bbox_class=class_name

        for label in config.labels:
            
            if(label in image_name_t):
            bbox_class=label

        for cate_idx,cate in enumerate(image):
            if(cate.size!=0):
                for bbox in cate:
                    if(bbox[4]<confidence_thr):
                        continue
                    bbox_dict={}
                    bbox_dict['score']=bbox[4]

    #                   print(bbox_dict['score'])

    #                         for i in bbox_dict['score']:
#                         f.write( str(bbox[4]) +'\n')
    #                     data = bbox_dict['score']
    #                     data.to_csv(config.WORK_DIR+'/'+'val_confidence.csv',index=False,colums=['confidence'])
                    bbox_dict['cate']=cate_idx
                    bbox_dict['bbox']=bbox[:4]
                    bbox_dict['image_name']=json_data['images'][image_idx]['file_name']
                    bbox_dict['image_class']=bbox_class
                    bbox_dict['image_class_idx']=config.labels2idx[bbox_class]
                    image_list.append(bbox_dict)
#                     print(bbox_dict)
         pr_list.append(image_list)
    #print(len(pr_list))
#     f.close()
    
    return pr_list  

#质心排序
class SpaceSorted:
    def getCenter(bbox):#获得一个bbox的中心点
        center=[bbox[0]+(bbox[2]-bbox[0])/2,bbox[1]+(bbox[3]-bbox[1])/2]
        return center
    
    def getDistance2(bbox):
        center = SpaceSorted.getCenter(bbox)
        return((abs((center[0]-256)**2+(center[1]-256)**2))**0.5)/362
    
    def getDistance3(bbox):
        center = SpaceSorted.getCenter(bbox)
        return((abs((center[0]-256)**2 + (center[1]-256)**2))**0.5)
    
    def getArea(bbox):#获得一个bbox的面积
        return (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
    
    def getMaxBbox(image):
        maxIdx=0
        maxStDtA=0
        if(str(type(image[0]))=="<class 'dict'>"):
            for idx,i in enumerate(image):
                if(i['score']*(362-SpaceSorted.getDistance3(i['bbox']))*SpaceSorted.getArea(i['bbox'])>maxStDtA):
                    maxStDtA=(362-SpaceSorted.getDistance3(i['bbox']))*i['score']*SpaceSorted.getArea(i['bbox'])
                    maxIdx=idx
            return image[maxIdx]
        else:
            for idx,i in enumerate(image):
                if(i[0]['score']*(362-SpaceSorted.getDistance3(i[0]['bbox'])*SpaceSorted.getArea(i[0]['bbox']))>maxStDtA):
                    maxStDtA=(362-SpaceSorted.getDistance3(i[0]['bbox']))*i[0]['score']*SpaceSorted.getArea(i[0]['bbox'])
                    maxIdx=idx
            return image[maxIdx][0]
    
    def getDistance(bbox1,bbox2):#获得两个bbox中心点间的距离
        center1=SpaceSorted.getCenter(bbox1)
        center2=SpaceSorted.getCenter(bbox2)
        return (abs((center2[0]-center1[0])**2+(center2[1]-center1[1])**2))**0.5
    
    def getDiagonal(bbox1):
        return ((bbox1[3]-bbox1[1])**2+(bbox1[2]-bbox1[0])**2)**0.5    

    def spaceSort(image):#最大的排前面，然后按照与最大的距离排序，但是不插空
        image_sorted=image.copy()
        max_bbox=SpaceSorted.getMaxBbox(image)
        for i in range(len(image)-1):
            for j in range(len(image)-1-i):
                if(SpaceSorted.getDistance(max_bbox['bbox'],image_sorted[j]['bbox'])>SpaceSorted.getDistance(max_bbox['bbox'],image_sorted[j+1]['bbox'])):
                    t=image_sorted[j]
                    image_sorted[j]=image_sorted[j+1]
                    image_sorted[j+1]=t
        return image_sorted
    
    def blankNum(bbox1, bbox2):
        # 输入两个bbox，判断要插几个空
        # 算法：两个bbox中心的距离减去bbox1的对角线/2，减去bbox2的对角线/2，上述结果除以小的bbox的对角线，四舍五入取整
        bbox1_diagonal = SpaceSorted.getDiagonal(bbox1)
        bbox2_diagonal = SpaceSorted.getDiagonal(bbox2)
        num_blank = SpaceSorted.getDistance(bbox1, bbox2) - (bbox1_diagonal + bbox2_diagonal) / 2
        num_blank = num_blank / min(SpaceSorted.getDiagonal(bbox1), SpaceSorted.getDiagonal(bbox2))
        num_blank = int(round(num_blank, 0))
        if (num_blank < 0):
            num_blank = 0
        return num_blank                        

    def spaceSortBlank(image):#插空算法
        image_sorted=SpaceSorted.spaceSort(image)
        vis=[] 
        for idx,i in enumerate(image_sorted):
            if(idx==0):#第一个直接添加到vis
                vis.append(i)
            else:
                #min_visited 距离当前i最近的bbox
                min_distance=99999
                for visited in vis:  
                    if(visited==[]):#插空后要跳过
                        continue
                    if(SpaceSorted.getDistance(visited['bbox'],i['bbox'])<min_distance):
                        min_distance=SpaceSorted.getDistance(visited['bbox'],i['bbox'])
                        min_distance_bbox=visited
                num_blank=SpaceSorted.blankNum(min_distance_bbox['bbox'],i['bbox'])
                for blank in range(num_blank):
                    vis.append([])
                vis.append(i)
        return vis

def get_list_max(t):#sort排序算法用
    return max(t)
def OneHot(score,cate,size):#分数，种类，长度
    t=[0.0 for i in range(size)]
    t[cate]=score
    return t

def OneHotMatrix(score,cate,size):
    t1=config.matrix[cate]
    t2=[0.0 for i in range(size)]
    t_sum_7=1-t1[cate]
    for i in range(size):
        t2[i]=(1-score)*t1[i]/t_sum_7
    t2[cate]=score
    return t2

def OneHotAdd(score,cate,list_):#往list_t中添加onehot编码
    list_[cate]=score
    return list_

def OneHot2(score,cate,size):#分数，种类，长度
    t=[0.0 for i in range(size)]
    t[cate]=score
    return t

def List2OneHot(image_list,fun,length,matrix=False,merge=False,onehot='confidence'):#将一个image的信息转为10*8的onehot fun表示排序方式。

    if (image_list==[]):
        return [[0.0 for i in range(8)] for j in range(length)]

    if(fun=='None' or fun=='confidence' or fun=='spaceOrdSort'):
        list_t=[]
        if(fun=='spaceOrdSort'):
            image_list=SpaceSort.spaceOrdSort(image_list)
        for bbox in image_list:
            if(matrix==True):
                list_t.append(OneHotMatrix(bbox['score'],bbox['cate'],8))
            elif(merge==True):
                t3=[0.0 for t0 in range(8)]
                #print(t3)
                for j in bbox:
                    #print(OneHotAdd(bbox['score'],bbox['cate'],t3)
                    #print(OneHotAdd(bbox['score'],bbox['cate'],t3)
                    t3=OneHotAdd(j['score'],j['cate'],t3)
                list_t.append(t3)
            else:
                list_t.append(OneHot(bbox['score'],bbox['cate'],8))
    if(fun=='confidence'):
        list_t.sort(key=get_list_max,reverse=True)
    elif(fun=='spaceSortBlank01'):
        list_t=[]
        image_list=SpaceSorted.spaceSortBlank(image_list)
        
        for bbox in image_list:
            if(bbox==[]):
                list_t.append([0.0 for i in range(8)])
            else:
                if(onehot=='confidence'):
                    list_t.append(OneHot(bbox['score'],bbox['cate'],8))
                elif(onehot=='1'):
                    list_t.append(OneHot(1,bbox['cate'],8))
    
    elif(fun=='spaceSortBlank'):
        list_t=[]
        max_bbox=SpaceSorted.getMaxBbox(image_list)
        max_bbox_area=SpaceSort.getArea(max_bbox['bbox'])
        image_list=SpaceSort.spaceSortBlank(image_list)
        
        for bbox in image_list:
            if(bbox==[]):
                list_t.append([0.0 for i in range(8)])
            else:
                bbox_area=SpaceSort.getArea(bbox['bbox'])
                if(onehot=='confidence'):
                    list_t.append(OneHot(bbox['score'],bbox['cate'],8))
                elif(onehot=='area'):
                    list_t.append(OneHot(bbox_area/max_bbox_area,bbox['cate'],8))
                elif(onehot=='1'):
                    list_t.append(OneHot(1,bbox['cate'],8))
                elif(onehot=='confidenceXarea'):
                    list_t.append(OneHot(bbox['score']*(bbox_area/max_bbox_area),bbox['cate'],8))
                #list_t.append(OneHot(bbox['score'],bbox['cate'],8))
                #list_t.append(OneHot(bbox['score']*bbox_area,bbox['cate'],8))
    elif(fun=='spaceSortMerge'):
        list_t=[]
        max_bbox=SpaceSort.getMaxBbox(image_list)
        max_bbox_area=SpaceSort.getArea(max_bbox['bbox'])
        image_list=SpaceSort.spaceSortMerge(image_list)
        for bbox in image_list:
            t3=[0.0 for t0 in range(8)]
            for j in bbox:
                bbox_area=SpaceSort.getArea(j['bbox'])
                if(onehot=='confidence'):
                    t3=OneHotAdd(j['score'],j['cate'],t3)
                elif(onehot=='area'):
                    t3=OneHotAdd(bbox_area/max_bbox_area,j['cate'],t3)
                elif(onehot=='confidenceXarea'):
                    t3=OneHotAdd(j['score']*(bbox_area/max_bbox_area),j['cate'],t3)
            list_t.append(t3)
                
    #print(list_t)
        
        
    
    if(len(list_t)<length):#固定二维长度
        for i in range(length-(len(list_t))):
            list_t.append([0.0 for j in range(8)])
        
    elif(len(list_t)>length):
        list_t=list_t[:length]
    
    
    
    return list_t

def Bbox_labels(image,max_l):
    bbox_labels=[]
    for i in image:
        if (i==[]):
            bbox_labels.append(0.0)
        else:
            if (i['cate']==i['gt_cate']):
                bbox_labels.append(1.0)
            else:
                bbox_labels.append(0.0)
    if (len(bbox_labels)>max_l):
        bbox_labels=bbox_labels[:max_l]
    else:
        for i in range(max_l-len(bbox_labels)):
            bbox_labels.append(0.0)
    return bbox_labels
class BEAUTY_bbox_Dataset(Dataset):
    def __init__(self,pr_list,fun,max_l,merge=False,onehot='confidence',multi_task='False'):
        super().__init__()
        self.pr_list = pr_list
        self.fun=fun
        self.max_l=max_l
        self.merge=merge
        self.onehot=onehot
        self.multi_task=multi_task

    def __len__(self):
        return len(self.pr_list)

    def __getitem__(self, idx):
        #print(idx)
        #print(len(self.pr_list))
        while(len(self.pr_list[idx])==0):
            idx+=1
        #print(idx)
        image=self.pr_list[idx]
        #print(image[0]['image_name'])
        #print(image)
        input_onehot=List2OneHot(image,self.fun,self.max_l,onehot=self.onehot)#************True记得放到config里面去
        input_onehot.reverse()
        #print(input_onehot)
        if(self.merge==True):
            label=image[0][0]['image_class_idx']
        else:
            label=image[0]['image_class_idx']
        #print(image)
        #print(input_onehot)
        input_onehot=torch.Tensor(input_onehot)
#         print(input_onehot)
        
#         if (self.multi_task=='True'):
#             bbox_label=Bbox_labels(image,self.max_l)
#             bbox_label=torch.Tensor(bbox_label)
#             return input_onehot,bbox_label,label

        return input_onehot,label

#位置编码
class PositionalEncoded(nn.Module):

    def __init__(self, d_model, max_len=8):
        super(PositionalEncoded, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe

#自注意力模块
class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 1, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 1, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x,att):
        #建筑类被之间的自适应加权
        if(att == 1):
            m_batchsize, C, width, height = x.size()
            proj_query = self.query_conv(x).view(m_batchsize, -1, width , height).permute(0,1,3,2)  # B X CX(N)
            proj_key = self.key_conv(x).view(m_batchsize, -1, width ,height)  # B X C x (*W*H)
            energy = torch.matmul(proj_query, proj_key)  # transpose check
            attention = self.softmax(energy)  # BX (N) X (N)
            proj_value = self.value_conv(x).view(m_batchsize, -1, width , height)  # B X C X N

            out = torch.matmul(proj_value, attention)
            out = out.view(m_batchsize, C, width, height)

            out = self.gamma * out + x
        #框与框之间的自适应加权
        elif(att == 2):
            m_batchsize, C, width, height = x.size()
            proj_query = self.query_conv(x).view(m_batchsize, -1, width , height).permute(0,1,3,2)  # B X CX(N)
            proj_key = self.key_conv(x).view(m_batchsize, -1, width ,height)  # B X C x (*W*H)
            energy = torch.matmul(proj_key,proj_query)  # transpose check
            attention = self.softmax(energy)  # BX (N) X (N)
            proj_value = self.value_conv(x).view(m_batchsize, -1, width , height)  # B X C X N
            
            out = torch.matmul( attention,proj_value)
            out = out.view(m_batchsize, C, width, height)

            out = self.gamma * out + x
        return out, attention


class Self_Attn2(nn.Module):
    """ Self attention Layer"""

    def __init__(self, ):
        super(Self_Attn2, self).__init__()

        self.query_conv = nn.Linear(8 * 10, 8 * 10)
        self.key_conv = nn.Linear(8 * 10, 8 * 10)
        self.value_conv = nn.Linear(8 * 10, 8 * 10)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        x = x.permute([1, 0, 2])
        m_batchsize, width, height = x.size()
        x= x.reshape(-1, 8*10)
        proj_query = self.query_conv(x).reshape(m_batchsize, 1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).reshape(m_batchsize, 1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).reshape(m_batchsize, 1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, width, height)

        x = x.reshape(m_batchsize, width, height)
        out = self.gamma * out + x
        return out, attention


class SA(nn.Module):
    def __init__(self, hidden_size, output_size):  # 输入，隐藏层单元，输出，隐藏层数，双向
        super(SA, self).__init__()

        self.hidden_size = hidden_size
        self.att1 = Self_Attn(1, 'relu') # 卷积
#         self.att1 = Self_Attn2() # 全连接
        self.out = nn.Linear(8 * hidden_size, output_size)

    def forward(self, input, tt):
        
        input = input.permute([1, 0, 2]) # 卷积        
        input2 = torch.tensor(config.Skip_Gram3_02_wordvector).to(config.device)
        input2 = input2.reshape(1,8,10)
        input = torch.matmul(input, input2)
        input = input.reshape([-1, 1, 8, 10]) # 卷积
#         positionalencoding = PositionalEncoded(10).to(config.device) 
#         input = positionalencoding(input)
        att = 2
        output1, hidden = self.att1(input,att)
        output2 = output1.reshape([-1, 80])

        output = self.out(output2)
        # output=F.softmax(output,dim=0)
        return output, hidden
    
#样本不平衡Loss
class MutilLabel_SmoothingFocalLoss(nn.Module):
    def __init__(self, class_num=4, alpha=0.2, gamma=2, use_alpha=False, size_average=True,smoothing=0.1):
        super(MutilLabel_SmoothingFocalLoss, self).__init__()
        self.class_num = class_num
        self.alpha = alpha
        self.gamma = gamma
        if use_alpha:
            self.alpha = torch.tensor(alpha).cuda()
            # self.alpha = torch.tensor(alpha).cuda()

        self.softmax = nn.Softmax(dim=1)
        self.use_alpha = use_alpha
        self.size_average = size_average
        self.smoothing = smoothing
        self.confidence = 0.9


    def forward(self, pred, target):
        prob = self.softmax(pred.view(-1,self.class_num))
        # print(prob)
        prob = prob.clamp(min=0.0001,max=1.0)  #防止log操作后，值过小

        #target_ = torch.zeros(target.size(0),self.class_num).cuda()
        target_ = torch.zeros(target.size(0),self.class_num).cuda()
        target_.scatter_(1, target.view(-1, 1).long(),1.)
        # print(target_)

        if self.use_alpha:
            batch_loss = - self.alpha.double() * torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()
        else:
            batch_loss = - torch.pow(1-prob,self.gamma).double() * prob.log().double() * target_.double()

        batch_loss = batch_loss.sum(dim=1)
        smooth_loss = -prob.log().double().mean()
        # print(smooth_loss)
        batch_loss = self.confidence * batch_loss + self.smoothing * smooth_loss

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
    
def evaluate_loss(data_iter, net,hidden,device, epoch):
    net.eval()
    l_sum, n,val_pred_sum= 0.0, 0,0
    with torch.no_grad():
        for bidx,(X, y) in enumerate(data_iter):
            X=X.float()
            X=X.permute(1,0,2)#[32, 10, 8]->[10,32,8]
            y=y.to(device)
            X=X.to(device)
            
            y_hat,hidden= net(X,hidden)
            loss = criterion(y_hat, y)
            
            l_sum+=loss
            #l_sum +=criterion(y_hat, y).item()
            n += X.size()[1]
            pred=y_hat.max(1, keepdim=True)[1].view(-1)
            pred_sum=pred.eq(y.view_as(pred)).sum().item()
            val_pred_sum+=pred_sum
    #print('val,epoch:{},pred:{},loss:{}'.format(epoch,val_pred_sum/n,l_sum/n))
    logger.info("[epoch {}][{}][end] val_loss={:.5f},val_acc:{:.5f}({}/{})".format\
    (epoch,'val',l_sum/(bidx+1),val_pred_sum/n,int(val_pred_sum),n))
    return l_sum / (bidx+1),val_pred_sum/n

def train(net,train_iter,val_iter,num_epoch,lr_period,lr_decay):
    optimizer = torch.optim.Adam(net.parameters(),lr=0.01)
    hidden=None
    Max_Acc=0.0
    Min_loss=9999999.9
    for epoch in range(num_epoch):
        net.train()
        n,train_l_sum,train_pred_sum=0,0,0
        if epoch > 0 and epoch % lr_period == 0:
            optimizer.param_groups[0]['lr']=optimizer.param_groups[0]['lr']*lr_decay
            
        for bidx, (X,y) in enumerate(train_iter):
            X=X.float()
            X=X.permute(1,0,2)#[32, 10, 8]->[10,32,8]
            X=X.to(config.device)
            y=y.to(config.device)
            if(hidden is not None):
                
                if isinstance (hidden, tuple): # LSTM, state:(h, c)  
                    hidden[0].to(config.device)
                    hidden[1].to(config.device)
                    hidden = (hidden[0].detach(), hidden[1].detach())
                else:   
                    hidden.to(config.device)
                    hidden = hidden.detach()
            
            optimizer.zero_grad()
            #print(str(bidx)+'---------------')
            #print(hidden)
            y_hat,hidden= net(X,hidden)
#             print(y_hat.size())
#             print(y.size())
#             print(X.size())
            loss = criterion(y_hat, y)
            #print(y_hat)
            #print(y)
            #print(loss.item())
            loss.backward()
            optimizer.step()
            #print(loss)
            pred=y_hat.max(1, keepdim=True)[1].view(-1)
            pred_sum=pred.eq(y.view_as(pred)).sum().item()
            #print(pred_sum)
            train_l_sum+=loss
            train_pred_sum+=pred_sum
            n+=X.size(1)
        #print("train,epoch:{},pred:{},loss:{}".format(epoch,train_pred_sum/n,train_l_sum/n))
        #print(str(train_pred_sum)+'/'+str(n))
        #print(train_pred_sum/n)
        if not os.path.exists('./params'):
            os.makedirs('./params')
        logger.info("[epoch {}][{}][end] train_loss={:.5f},train_acc={:.5f}({}/{})".format\
                    (epoch,'train',train_l_sum/(bidx+1),train_pred_sum/n,int(train_pred_sum),n))
        valid_loss,valid_acc=evaluate_loss(val_iter, net,hidden,config.device,epoch)
        if(valid_loss<Min_loss):
            Min_loss=valid_loss
            model_best=net
            #torch.save(net,'./params/'+MODEL_NAME+'_'+DATASET_NAME+'_best.pth')
           # torch.save(net,'./params/{}_BIDI-{}_NUMLAYERS-{}_HIDDENSIZE-{}_best.pth'.format\
                      # (RNN_TYPE,BIDIRECTIONAL,NUM_LAYERS,HIDDEN_SIZE))
            logger.info("[epoch {}][save_best_output_params]".format(epoch))
            #n+=y.size()[0]
            #print(n)

    return model_best

#test后，得出acc
def test_matrix(net,test_iter,device):
    net.eval()
    matrix=[[0 for j in range(4)] for i in range(4)]
    
    hidden=None
    sum=0
    n=0
    for bidx, (X,y) in enumerate(test_iter):
        X=X.float()
        X=X.permute(1,0,2)#[32, 10, 8]->[10,32,8]
        X=X.to(device)
        y=y.to(device)
        if(hidden is not None):

            if isinstance (hidden, tuple): # LSTM, state:(h, c)  
                hidden[0].to(device)
                hidden[1].to(device)
                hidden = (hidden[0].detach(), hidden[1].detach())
            else:   
                hidden.to(device)
                hidden = hidden.detach()
        y_hat,hidden= net(X,hidden)
        pred=y_hat.max(1, keepdim=True)[1].view(-1)

        pred=pred.cpu().numpy().tolist()
        y=y.cpu().numpy().tolist()
        for idx in range(len(y)):
            #print(idx)
            matrix[y[idx]][pred[idx]]+=1

        #print(y_hat)
        #print(y)
        #pred_sum=pred.eq(y.view_as(pred)).sum().item()
        #sum+=pred_sum
        #n+=len(y)
    return matrix

def F1(maxtir):
    sum_Pre=0
    sum_Re=0
    sum_all=0
    sum_TP=0
    for i in range(len(maxtir)):
        TP=maxtir[i][i]
        FP_and_TP=0
        FN_and_TP=0
        for j in range(len(maxtir)):
            sum_all+=maxtir[i][j]
            FN_and_TP+=maxtir[i][j]
            FP_and_TP+=maxtir[j][i]
        sum_TP+=TP
        if (FP_and_TP==0):
            sum_Pre+=0.0
        else:
            sum_Pre+=TP/FP_and_TP
        if FN_and_TP==0:
            sum_Re+=0.0
        else:
            sum_Re+=TP/FN_and_TP
        #print('Pression:'+str(TP/FP_and_TP))
        #print('recall:'+str(TP/FN_and_TP))
        #print('\n')
    sum_Pre/=4
    sum_Re/=4
    return {'Pression':sum_Pre,'Recall':sum_Re,'F1':(2*sum_Pre*sum_Re)/(sum_Pre+sum_Re),'Acc':sum_TP/sum_all}

if __name__ == '__main__':
    DETECTIONs=['cascade_rcnn_r101','cascade_rcnn_r50','faster_rcnn_r101','faster_rcnn_r50']
    FUNs=['None','Spatial_Embedding_SA']
    list_result={}
    logger=get_logger()
 
    logger.info('--------------------SLEDDing_embedding_self-attention--------------------')
    DETECTIONs=['cascade_rcnn_r101','cascade_rcnn_r50','faster_rcnn_r101','faster_rcnn_r50']
    FUNs=['Spatial_Embedding_SA','None']
    list_result1={}
    
    list_result1["[DETECTION={},FUN={}]".format('cascade_rcnn_r101','None')]=[]
    for b in range(5):
        logger.info("[DETECTION={},FUN={}]".format('cascade_rcnn_r101','None'))
        train_pr_list=get_pr_list(config.PATH_PR['cascade_rcnn_r101']+'results_train.pkl',config.PATH_GT_TRAIN)
        val_pr_list=get_pr_list(config.PATH_PR['cascade_rcnn_r101']+'results_val.pkl',config.PATH_GT_VAL)
        test_pr_list=get_pr_list(config.PATH_PR['cascade_rcnn_r101']+'results_test.pkl',config.PATH_GT_TEST)

        train_ds=BEAUTY_bbox_Dataset(train_pr_list,'None',8,onehot='confidence')
        val_ds=BEAUTY_bbox_Dataset(val_pr_list,'None',8,onehot='confidence')
        test_ds=BEAUTY_bbox_Dataset(test_pr_list,'None',8,onehot='confidence')

        train_iter=DataLoader(train_ds,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=5,drop_last=True)
        val_iter=DataLoader(val_ds,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=5,drop_last=True)
        test_iter=DataLoader(test_ds,batch_size=32,shuffle=False,num_workers=5,drop_last=True)
    
    
        lr_period, lr_decay= 5, 0.1
        criterion = MutilLabel_SmoothingFocalLoss()
        model=SA(10,4).to(config.device)
        model=train(model,train_iter,val_iter,20,lr_period,lr_decay)

        maxtir=test_matrix(model,test_iter,config.device)
        result=F1(maxtir)
        result['maxtir']=maxtir
        list_result1["[DETECTION={},FUN={}]".format('cascade_rcnn_r101','None')].append(result)
        logger.info(str(maxtir))
        logger.info(str(F1(maxtir))+'\n')
        list_result['co-occurrence']=list_result1
    with open(config.WORK_DIR+'/5_31_seed(8个框)_SA(kernel=1)', 'w+') as f:
        json.dump(list_result, f)
