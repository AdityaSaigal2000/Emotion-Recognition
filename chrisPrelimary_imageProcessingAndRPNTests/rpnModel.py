#help from: https://medium.com/@fractaldle/guide-to-build-faster-rcnn-in-pytorch-95b10c273439

import numpy as np
import torch.nn as nn
import torchvision

class SmallNet(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.name = "small"
        self.conv = nn.Conv2d(3, 5, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(5 * 7 * 7, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.pool(x)
        x = x.view(-1, 5 * 7 * 7)
        x = self.fc(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x

class RPNmodel(nn.Module):
    def __init__(self):
        super(SmallNet, self).__init__()
        self.name = "small"
        self.conv = nn.Conv2d(3, 5, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(5 * 7 * 7, 1)    
    #create dummy image and set volatile to be false

    dummy_img = torch.zeros((1, 3, 800, 800)).float()
    #print(dummy_img)

    #make the model
    model = torchvision.models.vgg16(pretrained=True)
    fe = list(model.features)
    #print(fe) # length is 15

    #pass the dummy image through the layers 
    req_features = []
    k = dummy_img.clone()
    for i in fe:
        k = i(k)
        if k.size()[2] < 800//16:
            break
        req_features.append(i)
        out_channels = k.size()[1]
    #print(len(req_features)) #30
    #print(out_channels) # 512


    #make the network
    faster_rcnn_fe_extractor = nn.Sequential(*req_features)
    out_map = faster_rcnn_fe_extractor(image)
    print(out_map.size())



    ###################The RPN
    #the actual RPN
    import torch.nn as nn
    mid_channels = 512
    in_channels = 512 # depends on the output feature map. in vgg 16 it is equal to 512
    n_anchor = 9 # Number of anchors at each location
    conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
    reg_layer = nn.Conv2d(mid_channels, n_anchor *4, 1, 1, 0)
    cls_layer = nn.Conv2d(mid_channels, n_anchor *2, 1, 1, 0) ## I will be going to use softmax here. you can equally use sigmoid if u replace 2 with 1.


    # conv sliding layer
    conv1.weight.data.normal_(0, 0.01)
    conv1.bias.data.zero_()
    # Regression layer
    reg_layer.weight.data.normal_(0, 0.01)
    reg_layer.bias.data.zero_()
    # classification layer
    cls_layer.weight.data.normal_(0, 0.01)
    cls_layer.bias.data.zero_()

    x = conv1(out_map) # out_map is obtained in section 1
    pred_anchor_locs = reg_layer(x)
    pred_cls_scores = cls_layer(x)
    #print(pred_cls_scores.shape, pred_anchor_locs.shape)
    #Out:
    #torch.Size([1, 18, 50, 50]) torch.Size([1, 36, 50, 50])

    pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
    print(pred_anchor_locs.shape)
    #Out: torch.Size([1, 22500, 4])
    pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()
    print(pred_cls_scores)
    #Out torch.Size([1, 50, 50, 18])
    objectness_score = pred_cls_scores.view(1, 50, 50, 9, 2)[:, :, :, :, 1].contiguous().view(1, -1)
    print(objectness_score.shape)
    #Out torch.Size([1, 22500])
    pred_cls_scores  = pred_cls_scores.view(1, -1, 2)
    print(pred_cls_scores.shape)
    # Out torch.size([1, 22500, 2])



