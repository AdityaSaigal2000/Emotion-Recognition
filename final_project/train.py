import os
import torch
from torch.autograd import Variable

import preprocess
from models.darknet_train import Darknet
from utils import weights_init_normal

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfgfile = "cfg/yolov3.cfg"
    weightsfile = "cfg/yolov3.weights"
    num_epochs = 100
    batch_size = 64

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    class_names = ['face']

    # Initiate model
    model = Darknet(cfgfile).to(device)
    model.apply(weights_init_normal)

    model.load_weights(weightsfile)

    # Get dataloader 
    imdb, roidb, ratio_list, ratio_index = preprocess.combined_roidb('train')
    train_size = len(roidb)
    sampler_batch = preprocess.sampler(train_size, batch_size)
    
    dataset = preprocess.roibatchLoader(roidb, ratio_list, ratio_index, batch_size, imdb.num_classes, training=True)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler_batch, num_workers=0)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    for epoch in range(num_epochs):
        model.train()
        for imgs, _, targets, _ in dataloader:

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)
    
            optimizer.zero_grad()
    
            loss = model(imgs, False, targets)
            print(loss.item())
    
            loss.backward()
            optimizer.step()
    
            model.seen += imgs.size(0)

        if epoch % 10 == 0:
            model.save_weights()
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)