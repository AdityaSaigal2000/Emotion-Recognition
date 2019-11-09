import torch
image = torch.zeros((1, 3, 800, 800)).float()

bbox = torch.FloatTensor([[20, 30, 400, 500], [300, 400, 500, 600]]) # [y1, x1, y2, x2] format
labels = torch.LongTensor([6, 8]) # 0 represents background
sub_sample = 16

##############################################
'''
#create dummy image and set volatile to be false
import torchvision
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
import torch.nn as nn
faster_rcnn_fe_extractor = nn.Sequential(*req_features)
out_map = faster_rcnn_fe_extractor(image)
print(out_map.size())
'''
############################################################3
#make the anchors in numpy
import numpy as np
ratios = [0.5, 1, 2]
anchor_scales = [8, 16, 32]

anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)

#print(anchor_base)

#make the anchor boundaries:
ctr_y = sub_sample / 2.
ctr_x = sub_sample / 2.

#print(ctr_y, ctr_x)
# Out: (8, 8)
for i in range(len(ratios)):
  for j in range(len(anchor_scales)):
    h = sub_sample * anchor_scales[j] * np.sqrt(ratios[i])
    w = sub_sample * anchor_scales[j] * np.sqrt(1./ ratios[i])

    index = i * len(anchor_scales) + j

    anchor_base[index, 0] = ctr_y - h / 2.
    anchor_base[index, 1] = ctr_x - w / 2.
    anchor_base[index, 2] = ctr_y + h / 2.
    anchor_base[index, 3] = ctr_x + w / 2.


#generating anchor points at the feature map locations
fe_size = (800//16)
ctr_x = np.arange(16, (fe_size+1) * 16, 16)
ctr_y = np.arange(16, (fe_size+1) * 16, 16)

ctr = []
index = 0
for x in range(len(ctr_x)):
    for y in range(len(ctr_y)):
        ctr += [[ctr_x[x] - 8, ctr_y[y] - 8]]
        index +=1

anchors = np.zeros((fe_size * fe_size * 9, 4))
index = 0
for c in ctr:
  ctr_y, ctr_x = c
  for i in range(len(ratios)):
    for j in range(len(anchor_scales)):
      h = sub_sample * anchor_scales[j] * np.sqrt(ratios[i])
      w = sub_sample * anchor_scales[j] * np.sqrt(1./ ratios[i])
      anchors[index, 0] = ctr_y - h / 2.
      anchors[index, 1] = ctr_x - w / 2.
      anchors[index, 2] = ctr_y + h / 2.
      anchors[index, 3] = ctr_x + w / 2.
      index += 1
print(anchors)
#Out: [22500, 4]




#########################################################
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


