import torch
import os
from tqdm import tqdm
import numpy as np
import argparse
import cv2 as cv

import sys
sys.path.append('../../')
from pretrain_networks.visual_frontend import VisualFrontend

def write_npy(fname, file):
    fdir = os.path.dirname(fname)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    np.save(fname,file)

def main(args):
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")

    #declaring the visual frontend module
    vf = VisualFrontend()
    vf.load_state_dict(torch.load('../../pretrain_networks/visual_frontend.pt', map_location=device))
    vf.to(device)

    roiSize = 112
    normMean_pre = 0.52411
    normStd_pre = 0.22226

    normMean = 0.4161
    normStd = 0.1688

    utts = []
    for path, dirs ,files in os.walk(args.visual_frame_direc):
        for filename in files:
            if filename[-4:] =='.npy' and filename[0] != '.':
                ln = path + '/' + filename
                utts.append(ln)


    for utt in tqdm(utts,desc = "Generating visual embedding"):
        sav_path = utt.replace('/image/','/lip/')
        grayed = np.load(utt)

        if os.path.exists(sav_path):
            continue

        roi = grayed[:,int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2)), int(roiSize-(roiSize/2)):int(roiSize+(roiSize/2))]
        
        # save small version of image, optional
        sav_path_img_small = utt.replace('/image/','/image_small/')
        write_npy(sav_path_img_small, roi)

        roi = roi * normStd_pre + normMean_pre
        inp = np.expand_dims(roi, axis=[1,2])
        inp = (inp - normMean)/normStd
        inputBatch = torch.from_numpy(inp)
        inputBatch = (inputBatch.float()).to(device)
        vf.eval()
        with torch.no_grad():
            outputBatch = vf(inputBatch)
        out = torch.squeeze(outputBatch, dim=1)
        out = out.cpu().numpy()
        # print(out.shape)
        write_npy(sav_path, out)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LRS3 dataset')
    parser.add_argument('--lip_embedding_direc',default ='/home/panzexu/datasets/iemocap/uss/visual_embedding/lip/', type=str)
    parser.add_argument('--visual_frame_direc',default ='/home/panzexu/datasets/iemocap/uss/visual_embedding/image/', type=str)
    args = parser.parse_args()
    main(args)