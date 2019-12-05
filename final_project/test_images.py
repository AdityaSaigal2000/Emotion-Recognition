from __future__ import division
import time
import torch 
from torch.autograd import Variable
import numpy as np
import cv2 
from utils import write_results, softmax
import os 
import os.path as osp
from models.darknet import Darknet
from preprocess import prep_image
import pandas as pd
import random 
import pickle as pkl

from models.emotion_classifier import Model
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]
net = Model()

def write(x, batches, results):
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        img = results[int(x[0])]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi = gray[c1[1]:c2[1], c1[0]:c2[0]]
        roi = cv2.resize(roi, (48, 48))
        roi = (roi.astype("float") / 127.5) - 1.0
        roi = np.expand_dims(roi, axis=0)
        roi = torch.FloatTensor(roi).unsqueeze(0)
        preds = net(roi)[0].detach().numpy()
        emotion_probability = np.max(softmax(preds))
        emotion = EMOTIONS[preds.argmax()]
        label = "{}: {:.2f}%".format(emotion, emotion_probability * 100)
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2,color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2,color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
        return img

if __name__ ==  '__main__':
    
    scales = "1,2,3"

    images = './input'
    batch_size = 1
    confidence = 0.5
    nms_thesh = 0.4
    start = 0

    CUDA = torch.cuda.is_available()

    num_classes = 1
    #classes = load_classes('data/face.names')
    classes = ['face']

    #Set up the neural network
    print("Loading network.....")
    model = Darknet("./cfg/yolov3.cfg")
    model.load_weights("./cfg/yolov3.weights")
    
    state = torch.load('cfg/model_params.chkpt', map_location=torch.device('cpu'))
    net.load_state_dict(state)
    print("Network successfully loaded")
    
    model.net_info["height"] = "416"
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    #If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()
    
    
    #Set the model in evaluation mode
    model.eval()
    
    read_dir = time.time()
    #Detection phase
    try:
        imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images) if os.path.splitext(img)[1] == '.png' or os.path.splitext(img)[1] =='.jpeg' or os.path.splitext(img)[1] =='.jpg']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), images))
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(images))
        exit()
        
    if not os.path.exists("./output"):
        os.makedirs("./output")
        
    load_batch = time.time()
    
    batches = list(map(prep_image, imlist, [inp_dim for x in range(len(imlist))]))
    im_batches = [x[0] for x in batches]
    orig_ims = [x[1] for x in batches]
    im_dim_list = [x[2] for x in batches]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)
    
    
    
    if CUDA:
        im_dim_list = im_dim_list.cuda()
    
    leftover = 0
    
    if (len(im_dim_list) % batch_size):
        leftover = 1
        
        
    if batch_size != 1:
        num_batches = len(imlist) // batch_size + leftover            
        im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                            len(im_batches))]))  for i in range(num_batches)]        


    i = 0
    

    written = False
    
    start_det_loop = time.time()
    
    objs = {}
    
    for batch in im_batches:
        #load the image 
        start = time.time()
        if CUDA:
            batch = batch.cuda()
        
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)
        
        prediction = write_results(prediction, confidence, num_classes, nms = True, nms_conf = nms_thesh)
        

        if prediction.shape[1] == 7:
            i += 1
            continue

        end = time.time()        

        prediction[:,0] += i*batch_size
    
        if not written:
            output = prediction
            written = 1
        else:
            output = torch.cat((output,prediction))
            
        
        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
            print("{0:20s} predicted in {1:6.3f} seconds".format(image.split("/")[-1], (end - start)/batch_size))
            print("{0:20s} {1:s}".format("Objects Detected:", " ".join(objs)))
            print("----------------------------------------------------------")
        i += 1

        
        if CUDA:
            torch.cuda.synchronize()
    
    try:
        output
    except NameError:
        print("No detections were made")
        exit()
        
    im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())
    
    scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)
    
    
    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2
    
    
    
    output[:,1:5] /= scaling_factor
    
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
        
        
    output_recast = time.time()
    
    
    class_load = time.time()

    colors = pkl.load(open("pallete", "rb"))
    
    
    draw = time.time()
           
    list(map(lambda x: write(x, im_batches, orig_ims), output))
      
    det_names = pd.Series(imlist).apply(lambda x: "{}/det_{}".format("./output",x.split("/")[-1]))
    
    list(map(cv2.imwrite, det_names, orig_ims))
    
    end = time.time()
    
    print()
    print("SUMMARY")
    print("----------------------------------------------------------")
    print("{:25s}: {}".format("Task", "Time Taken (in seconds)"))
    print()
    print("{:25s}: {:2.3f}".format("Reading addresses", load_batch - read_dir))
    print("{:25s}: {:2.3f}".format("Loading batch", start_det_loop - load_batch))
    print("{:25s}: {:2.3f}".format("Detection (" + str(len(imlist)) +  " images)", output_recast - start_det_loop))
    print("{:25s}: {:2.3f}".format("Output Processing", class_load - output_recast))
    print("{:25s}: {:2.3f}".format("Drawing Boxes", end - draw))
    print("{:25s}: {:2.3f}".format("Average time_per_img", (end - load_batch)/len(imlist)))
    print("----------------------------------------------------------")

    
    torch.cuda.empty_cache()
    