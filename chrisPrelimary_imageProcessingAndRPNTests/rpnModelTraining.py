#code based off of tutorial from: https://medium.com/@fractaldle/guide-to-build-faster-rcnn-in-pytorch-95b10c273439
import torch
import numpy as np
import torch.nn as nn
import torchvision
import torch.nn.functional as F


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
        return ROIs, pred_anchor_locs, pred_cls_scores, objectness_score, pred_cls_scores 

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

#this training loss function based on the same tutorial as mentioned above
def training(imageWidth, featureWidth, rpnModel, trainingFeatureMapInput, boundaryBoxGroundTruths):
    #TRAINING STUFF:
    anchors = generateAnchors(featureWidth, imageWidth)
    #creating arbitrary bounding boxes
    bbox = np.asarray([[20, 30, 80, 90], [100, 100, 150, 150]], dtype=np.float32) # [y1, x1, y2, x2] format
    labels = np.asarray([1, 2], dtype=np.int8) # 0 represents background

    #finding the index of valid anchor boxes
    index_inside = np.where(
            (anchors[:, 0] >= 0) &
            (anchors[:, 1] >= 0) &
            (anchors[:, 2] <= imageWidth) &
            (anchors[:, 3] <= imageWidth)
        )[0]
    #print(index_inside.shape)
    #Out: (8940,)
    #create array with valid anchor boxes
    valid_anchor_boxes = anchors[index_inside]
    #print(valid_anchor_boxes.shape)
    #Out = (8940, 4)

    #create an empty label array with inside_index shape and fill with -1
    label = np.empty((len(index_inside), ), dtype=np.int32)
    label.fill(-1)
    #print(label.shape)
    #Out = (8940, )

    #Create the IOU ARRAY ious
    ious = np.empty((len(valid_anchor_boxes), len(bbox)), dtype=np.float32)#THIS WAS WRONG CHANGED IT TO LEN(BBOX)
    ious.fill(0)
    #print(bbox)
    for num1, anchor_locations in enumerate(valid_anchor_boxes):
        ya1, xa1, ya2, xa2 = anchor_locations
        anchor_area = (ya2 - ya1) * (xa2 - xa1)
        for num2, boundary_locations in enumerate(bbox):
            yb1, xb1, yb2, xb2 = boundary_locations
            box_area = (yb2- yb1) * (xb2 - xb1)
            inter_x1 = max([xb1, xa1])
            inter_y1 = max([yb1, ya1])
            inter_x2 = min([xb2, xa2])
            inter_y2 = min([yb2, ya2])
            if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
                iter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
                iou = iter_area / (anchor_area+ box_area - iter_area)            
            else:
                iou = 0.
            ious[num1, num2] = iou
    #print(ious.shape)
    #Out: [22500, 2]


    #the highest iou for each gt_box and its corresponding anchor box    
    gt_argmax_ious = ious.argmax(axis=0)
    #print(gt_argmax_ious)#its corresponding anchor box  
    gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
    #print(gt_max_ious)#the highest iou for each gt_box
    # Out:
    # [2262 5620]
    # [0.68130493 0.61035156]
    
    # the highest iou for each anchor box and its corresponding ground truth box
    argmax_ious = ious.argmax(axis=1)#its corresponding ground truth box
    #print(argmax_ious.shape)#
    #print(argmax_ious)
    max_ious = ious[np.arange(len(index_inside)), argmax_ious]#the maximum ious for each anchor box
    #print(max_ious)
    # Out:
    # (22500,)
    # [0, 1, 0, ..., 1, 0, 0]
    # [0.06811669 0.07083762 0.07083762 ... 0.         0.         0.        ]

    #finding the anchor boxes which have the highest ios in either bounding box
    gt_argmax_ious = np.where(ious == gt_max_ious)[0]########
    #print(gt_argmax_ious)
    # Out:
    # [2262, 2508, 5620, 5628, 5636, 5644, 5866, 5874, 5882, 5890, 6112,
    #        6120, 6128, 6136, 6358, 6366, 6374, 6382]

    #set up threshold variables
    pos_iou_threshold  = 0.7
    neg_iou_threshold = 0.3

    #assign the labels for whether each anchor box matches well with a ground truth box
    #Assign negitive label (0) to all the anchor boxes which have max_iou less than negitive threshold
    label[max_ious < neg_iou_threshold] = 0
    #Assign positive label (1) to all the anchor boxes which have highest IoU overlap with a ground-truth box
    label[gt_argmax_ious] = 1
    #Assign positive label (1) to all the anchor boxes which have max_iou greater than positive threshold 
    label[max_ious >= pos_iou_threshold] = 1

    ####THE ACTUAL TRAINING
    #setting up some variable
    pos_ratio = 0.5#the ratio of positive and negative labelled samples
    n_sample = 256 #the number of anchors to sample from the image

    n_pos = int(pos_ratio * n_sample)#the total number of positive samples
    
    #randomly sample n_pos samples from the positive labels and ignore (-1) the remaining ones
    #if there are less than n_pos samples, then it will randomly sample (n_sample — n_pos) negitive samples (0) and assign ignore label to the remaining anchor boxes
    #the positive sampling
    pos_index = np.where(label == 1)[0]
    if len(pos_index) > n_pos:
        disable_index = np.random.choice(pos_index, size=(len(pos_index) - n_pos), replace=False)
        label[disable_index] = -1
    n_neg = n_sample * np.sum(label == 1)
    #the negative sampling
    neg_index = np.where(label == 0)[0]
    if len(neg_index) > n_neg:
        disable_index = np.random.choice(neg_index, size=(len(neg_index) - n_neg), replace = False)
        label[disable_index] = -1

    #find the locations of the groundtruth object which has max_iou for each anchor
    max_iou_bbox = bbox[argmax_ious]
    #print(max_iou_bbox)

    #convert the y1, x1, y2, x2 format of valid anchor boxes and associated ground truth boxes with max iou to ctr_y, ctr_x , h, w format
    height = valid_anchor_boxes[:, 2] - valid_anchor_boxes[:, 0]
    width = valid_anchor_boxes[:, 3] - valid_anchor_boxes[:, 1]
    ctr_y = valid_anchor_boxes[:, 0] + 0.5 * height
    ctr_x = valid_anchor_boxes[:, 1] + 0.5 * width
    
    base_height = max_iou_bbox[:, 2] - max_iou_bbox[:, 0]
    base_width = max_iou_bbox[:, 3] - max_iou_bbox[:, 1]
    base_ctr_y = max_iou_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = max_iou_bbox[:, 1] + 0.5 * base_width

    #Use the formulas mentioned in the article to assign ground truth to anchor boxes (ie find the ideal output values of the rpn given the anchor box locations and ground truth locations)
    eps = np.finfo(height.dtype).eps
    height = np.maximum(height, eps)
    width = np.maximum(width, eps)
    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = np.log(base_height / height)
    dw = np.log(base_width / width)
    anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()
    #print(anchor_locs)
    #Out:
    # [[ 0.5855727   2.3091455   0.7415673   1.647276  ]
    #  [ 0.49718437  2.3091455   0.7415673   1.647276  ]
    #  [ 0.40879607  2.3091455   0.7415673   1.647276  ]
    #  ...
    #  [-2.50802    -5.292254    0.7415677   1.6472763 ]
    #  [-2.5964084  -5.292254    0.7415677   1.6472763 ]
    #  [-2.6847968  -5.292254    0.7415677   1.6472763 ]]

    #find the final anchor labels
    anchor_labels = np.empty((len(anchors),), dtype=label.dtype)
    anchor_labels.fill(-1)
    anchor_labels[index_inside] = label
    #find the final anchor locations
    anchor_locations = np.empty((len(anchors),) + anchors.shape[1:], dtype=anchor_locs.dtype)
    anchor_locations.fill(0)
    anchor_locations[index_inside, :] = anchor_locs


    ### THE ACTUAL TRAINING... AKA COMPUTE THE LOSS
    ROIs, pred_anchor_locs, pred_cls_scores, objectness_score, pred_cls_scores = rpnModel(trainingFeatureMapInput)

    '''
    print(pred_anchor_locs.shape)
    print(pred_cls_scores.shape)
    print(anchor_locations.shape)
    print(anchor_labels.shape)
    '''

    #some minor rearranging
    rpn_loc = pred_anchor_locs[0]
    rpn_score = pred_cls_scores[0]
    gt_rpn_loc = torch.from_numpy(anchor_locations)
    gt_rpn_score = torch.from_numpy(anchor_labels)
    #print(rpn_loc.shape, rpn_score.shape, gt_rpn_loc.shape, gt_rpn_score.shape)


    #calculate the classification loss
    rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_score.long(), ignore_index = -1)
    #print(rpn_cls_loss)
    #Out:
    # Variable containing:
    #  0.6940
    # [torch.FloatTensor of size 1]
    #Out
    # torch.Size([12321, 4]) torch.Size([12321, 2]) torch.Size([12321, 4]) torch.Size([12321])

    #Regression loss is applied to the bounding boxes which have positive labels
    pos = gt_rpn_score > 0
    mask = pos.unsqueeze(1).expand_as(rpn_loc)
    #print(mask.shape)
    #Out:
    # torch.Size(12321, 4)

    #select positive label bounding boxes
    mask_loc_preds = rpn_loc[mask].view(-1, 4)
    mask_loc_targets = gt_rpn_loc[mask].view(-1, 4)
    #print(mask_loc_preds.shape, mask_loc_preds.shape)
    #Out:
    # torch.Size([6, 4]) torch.Size([6, 4])

    #find the location (regression) loss
    x = torch.abs(mask_loc_targets.double() - mask_loc_preds.double())
    rpn_loc_loss = ((x < 1).double() * 0.5 * x**2) + ((x >= 1).double() * (x-0.5))
    #print(rpn_loc_loss.sum())
    #Out:
    # Variable containing:
    #  0.3826
    # [torch.FloatTensor of size 1]

    #find total loss. class loss is applied on all the bounding boxes and regression loss is applied only positive bounding box
    rpn_lambda = 10.
    N_reg = (gt_rpn_score >0).double().sum()
    rpn_loc_loss = rpn_loc_loss.sum() / N_reg
    rpn_loss = rpn_cls_loss + (rpn_lambda * rpn_loc_loss)
    #print(rpn_loss)
    #Out:0.00248
    return rpn_loss

'''
def testingRaw():
    bbox = np.asarray([[20, 30, 400, 500], [300, 400, 500, 600]], dtype=np.float32) # [y1, x1, y2, x2] format
    labels = np.asarray([6, 8], dtype=np.int8) # 0 represents background

    anchors = generateAnchors(50, 800)

    #find the indexes of valid anchor boxes and create an array with these indexes
    index_inside = np.where(
        (anchors[:, 0] >= 0) &
        (anchors[:, 1] >= 0) &
        (anchors[:, 2] <= 800) &
        (anchors[:, 3] <= 800)
    )[0]
    print(index_inside.shape)
    #Out: (8940,)
    #create array with valid anchor boxes
    valid_anchor_boxes = anchors[index_inside]
    print(valid_anchor_boxes.shape)
    #Out = (8940, 4)

    label = np.empty((len(index_inside), ), dtype=np.int32)
    label.fill(-1)
    print(label.shape)
    #Out = (8940, )

    #Create the IOU ARRAY ious
    ious = np.empty((len(valid_anchor_boxes), 2), dtype=np.float32)
    ious.fill(0)
    print(bbox)
    for num1, i in enumerate(valid_anchor_boxes):
        ya1, xa1, ya2, xa2 = i  
        anchor_area = (ya2 - ya1) * (xa2 - xa1)
        for num2, j in enumerate(bbox):
            yb1, xb1, yb2, xb2 = j
            box_area = (yb2- yb1) * (xb2 - xb1)
            inter_x1 = max([xb1, xa1])
            inter_y1 = max([yb1, ya1])
            inter_x2 = min([xb2, xa2])
            inter_y2 = min([yb2, ya2])
            if (inter_x1 < inter_x2) and (inter_y1 < inter_y2):
                iter_area = (inter_y2 - inter_y1) * \
    (inter_x2 - inter_x1)
                iou = iter_area / \
    (anchor_area+ box_area - iter_area)            
            else:
                iou = 0.
            ious[num1, num2] = iou
    print(ious.shape)
    #Out: [22500, 2]
'''
