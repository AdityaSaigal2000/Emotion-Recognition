from __future__ import division
import time
import torch 
from torch.autograd import Variable
import numpy as np
import cv2 
from utils import softmax, write_results
from models.darknet import Darknet
from preprocess import prep_image_vid
from models.emotion_classifier import Model
import random 
import pickle as pkl

EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]
net = Model()

def write(x, img):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    
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
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img


if __name__ == '__main__':
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "cfg/yolov3.weights"

    confidence = 0.25
    nms_thesh = 0.4
    start = 0
    CUDA = torch.cuda.is_available()
    
    state = torch.load('cfg/model_params.chkpt', map_location=torch.device('cpu'))
    net.load_state_dict(state)
    
    num_classes = 1
    bbox_attrs = 5 + num_classes
    
    model = Darknet(cfgfile)
    model.load_weights(weightsfile)
    
    model.net_info["height"] = "160"
    inp_dim = int(model.net_info["height"])
    
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()
            
    model.eval()
    
    videofile = 'video.avi'
    
    cap = cv2.VideoCapture(0)
    
    assert cap.isOpened(), 'Cannot capture source'
    
    frames = 0
    start = time.time()    
    while cap.isOpened():
        
        ret, frame = cap.read()
        if ret:
            
            img, orig_im, dim = prep_image_vid(frame, inp_dim)
      
            if CUDA:
                img = img.cuda()
            
            
            output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)

            if output.shape[1] == 7:
                frames += 1
                print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                continue
        
            output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
            
            output[:,[1,3]] *= frame.shape[1]
            output[:,[2,4]] *= frame.shape[0]
            
            classes = ['face']
            colors = pkl.load(open("pallete", "rb"))
            
            list(map(lambda x: write(x, orig_im), output))
            
            
            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

            
        else:
            break
    

    
    

