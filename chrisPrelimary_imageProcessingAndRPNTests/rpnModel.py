#code based off of tutorial from: https://medium.com/@fractaldle/guide-to-build-faster-rcnn-in-pytorch-95b10c273439
import torch
import numpy as np
import torch.nn as nn
import torchvision



class RPNFeatureExtractor(nn.Module):
    def __init__(self):
        super(RPNFeatureExtractor, self).__init__()
        self.name = "rpnFeatureExtractor"
        
        #use a dummy image to test when the output size of vgg network is below 50 and eliminate the rest
        #of the layers
        dummy_img = torch.zeros((1, 3, 210, 210)).float()#NOTE 210 is the size of the input image here
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
            if k.size()[2] < 50:
                break#trim off the remaining features with the incorrect output size
            req_features.append(i)#in this case it become 52*52 image with 256 input
            #out_channels = k.size()[1]
        #print(len(req_features)) #30
        #print(out_channels) # 512

        #make the network
        self.faster_rcnn_fe_extractor = nn.Sequential(*req_features)

    def forward(self, x):
        return self.faster_rcnn_fe_extractor(x)

class RPNmodel(nn.Module):
    def __init__(self):
        super(RPNmodel, self).__init__()
        self.name = "rpn"
        self.featureWidth = 52
        self.featureChannels = 256
        self.imageWidth = 210 #the original image width
        #help from: https://medium.com/@fractaldle/guide-to-build-faster-rcnn-in-pytorch-95b10c273439
        self.mid_channels = 256 #this was originally 256
        self.in_channels = self.featureChannels# Note this used to be 512 but the features are now 256 layers # for the output channels of vgg 16 feature extractor this is equal to 512
        self.n_anchor = 9 # Number of anchors at each location
        self.conv1 = nn.Conv2d(self.in_channels, self.mid_channels, 3, 1, 1)
        self.reg_layer = nn.Conv2d(self.mid_channels, self.n_anchor*4, 1, 1, 0)
        self.cls_layer = nn.Conv2d(self.mid_channels, self.n_anchor*2, 1, 1, 0) ## I will be going to use softmax here. you can equally use sigmoid if u replace 2 with 1.


        # conv sliding layer
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv1.bias.data.zero_()
        # Regression layer
        self.reg_layer.weight.data.normal_(0, 0.01)
        self.reg_layer.bias.data.zero_()
        # classification layer
        self.cls_layer.weight.data.normal_(0, 0.01)
        self.cls_layer.bias.data.zero_()

    def forward(self, x):#x is the feature map
        x = self.conv1(x)
        pred_anchor_locs = self.reg_layer(x)
        pred_cls_scores = self.cls_layer(x)
        #print(pred_cls_scores.shape, pred_anchor_locs.shape)
        #Out:
        #torch.Size([1, 18, 50, 50]) torch.Size([1, 36, 50, 50])

        pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
        #print(pred_anchor_locs.shape)
        #Out: torch.Size([1, 22500, 4])
        pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()
        #print(pred_cls_scores)
        #Out torch.Size([1, 50, 50, 18])
        objectness_score = pred_cls_scores.view(1, self.featureWidth, self.featureWidth, 9, 2)[:, :, :, :, 1].contiguous().view(1, -1)
        #note that this used to be 50 by 50 but now the feature maps are 52 by 52
        #print(objectness_score.shape)
        #Out torch.Size([1, 22500])
        pred_cls_scores  = pred_cls_scores.view(1, -1, 2)
        #print(pred_cls_scores.shape)
        # Out torch.size([1, 22500, 2])
        ROIs = proposeRegions(pred_anchor_locs, objectness_score, self.imageWidth, self.featureWidth)
        return ROIs

def proposeRegions(pred_anchor_locs, objectness_score, imageWidth, featureWidth):
    anchors = generateAnchors(featureWidth, imageWidth)
    nms_thresh = 0.7
    n_train_pre_nms = 12000
    n_train_post_nms = 2000
    n_test_pre_nms = 6000
    n_test_post_nms = 300
    min_size = 16

    #Convert anchors format from y1, x1, y2, x2 to ctr_x, ctr_y, h, w
    anc_height = anchors[:, 2] - anchors[:, 0]
    anc_width = anchors[:, 3] - anchors[:, 1]
    anc_ctr_y = anchors[:, 0] + 0.5 * anc_height
    anc_ctr_x = anchors[:, 1] + 0.5 * anc_width

    #Convert predictions locs using above formulas
    pred_anchor_locs_numpy = pred_anchor_locs[0].data.numpy()
    objectness_score_numpy = objectness_score[0].data.numpy()
    dy = pred_anchor_locs_numpy[:, 0::4]
    dx = pred_anchor_locs_numpy[:, 1::4]
    dh = pred_anchor_locs_numpy[:, 2::4]
    dw = pred_anchor_locs_numpy[:, 3::4]
    ctr_y = dy * anc_height[:, np.newaxis] + anc_ctr_y[:, np.newaxis]
    ctr_x = dx * anc_width[:, np.newaxis] + anc_ctr_x[:, np.newaxis]
    h = np.exp(dh) * anc_height[:, np.newaxis]
    w = np.exp(dw) * anc_width[:, np.newaxis]

    #convert [ctr_x, ctr_y, h, w] to [y1, x1, y2, x2] format
    roi = np.zeros(pred_anchor_locs_numpy.shape, dtype=pred_anchor_locs_numpy.dtype)
    roi[:, 0::4] = ctr_y - 0.5 * h
    roi[:, 1::4] = ctr_x - 0.5 * w
    roi[:, 2::4] = ctr_y + 0.5 * h
    roi[:, 3::4] = ctr_x + 0.5 * w

    #clip the predicted boxes to the image
    img_size = (imageWidth, imageWidth) #Image size
    roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, img_size[0])
    roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, img_size[1])
    #print(roi)
    ######
    #Remove predicted boxes with either height or width < threshold
    hs = roi[:, 2] - roi[:, 0]
    ws = roi[:, 3] - roi[:, 1]
    keep = np.where((hs >= min_size) & (ws >= min_size))[0]
    roi = roi[keep, :]
    score = objectness_score_numpy[keep]
    #print(score.shape)
    #Out:
    ##(22500, ) all the boxes have minimum size of 16

    #I THINK THIS IS AN ERROR IN THE TUTORIAL... RAVEL AND THEN ARGSORT PRODUCES INDICES GREATER THAN THE LENGTH OF THE ORIGINAL ARRAY
    #Sort all (proposal, score) pairs by score from highest to lowest
    order = score.ravel().argsort()[::-1]
    #DOING THIS INSTEAD:
    #for i in score:
    #print(order)

    #order = order[order < len(roi)]#remove indices that are larger than the length of the roi array

    #Take top pre_nms_topN
    #if the length of roi is less than the length of pre_nms then use it to avoid an error
    order = order[:min(n_train_pre_nms,len(roi), len(order))]#THIS CAN BE SWITCHED TO TEST INSTEAD OF TRAIN
   
    roi = roi[order, :]
    #print(roi.shape)
    #print(roi)

    #####
    #
    y1 = roi[:, 0]
    x1 = roi[:, 1]
    y2 = roi[:, 2]
    x2 = roi[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #order = score.argsort()[::-1]#commented this from the tutorial
    keep = []
    while order.size > 0:
        i = order[0]
        keep += [i]#NOTE THAT THE TUTORIAL HAS A MISTAKE HERE IT FORGOT TO ADD TO THE keep ARRAY
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds + 1]
        

    keep = keep[:min(n_train_post_nms, len(roi), len(keep))] # while training/testing , use accordingly
    roi = roi[keep] # the final region proposals
    return roi

def generateAnchors(featureWidth, imageWidth):
    sub_sample = int(imageWidth/featureWidth)#sub_sample is the ratio of the image width to the feature map width

    ratios = [0.5, 1, 2]
    anchor_scales = [4, 8, 16]
    '''
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

    '''
    #generating anchor points at the feature map locations
    #fe_size = (800//16)
    ctr_x = np.arange(sub_sample, (featureWidth+1) * sub_sample, sub_sample)
    ctr_y = np.arange(sub_sample, (featureWidth+1) * sub_sample, sub_sample)

    ctr = []
    index = 0
    for x in range(len(ctr_x)):
        for y in range(len(ctr_y)):
            ctr += [[ctr_x[x] - int(sub_sample/2), ctr_y[y] - int(sub_sample/2)]]
            index +=1

    anchors = np.zeros((featureWidth * featureWidth * len(ratios)*len(anchor_scales), 4))
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
    #print(anchors)
    return anchors