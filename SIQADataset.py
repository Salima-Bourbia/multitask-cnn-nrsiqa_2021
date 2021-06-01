import torch
import os
from torch.utils.data import Dataset
from scipy.signal import convolve2d
from PIL import Image
import numpy as np
from torchvision.transforms.functional import to_tensor
import cv2 as cv
import torchvision
from pathlib import Path




def gray_loader(path):
    return Image.open(path).convert('L')

def LocalNormalization(patch, P=7, Q=7, C=1):
    kernel = np.ones((P, Q)) / (P * Q)
    patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
    patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
    patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
    patch_ln = torch.from_numpy((patch - patch_mean) / patch_std).float().unsqueeze(0)
    return patch_ln

def CropPatches(image, patch_size=32, stride=24):
    w, h = image.size
    patches = ()
    for i in range(0, h-patch_size, stride):
        for j in range(0, w-patch_size, stride):
            patch = to_tensor(image.crop((j, i, j+patch_size, i+patch_size)))
            patch = LocalNormalization(patch[0].numpy())
            patches = patches + (patch,)
    return patches


class SIQADataset(Dataset):
    def __init__(self, dataset, config, index, status):
        self.gray_loader = gray_loader
        im_dirR = config[dataset]['im_dirR']
        im_dirL = config[dataset]['im_dirL']
        self.patch_size = config['patch_size']
        self.stride = config['stride']

        test_ratio = config['test_ratio']
        train_ratio = config['train_ratio']
        trainindex = index[:int(train_ratio * len(index))]
        testindex = index[int((1 - test_ratio) * len(index)):]
        train_index, val_index, test_index = [], [], []

        ref_ids = []
        for line0 in open("./data_copule/ref_ids.txt", "r"):
            line0 = float(line0[:-1])
            ref_ids.append(line0)
        ref_ids = np.array(ref_ids)

        for i in range(len(ref_ids)):
            if(ref_ids[i] in trainindex):
                train_index.append(i)
            elif(ref_ids[i] in testindex):
                test_index.append(i)
            else:
                val_index.append(i)


        if status == 'train':
            self.index = train_index
            print("# Train Images len: {}".format(len(self.index)))
            print('Training Ref Index:')
            print(trainindex)
        if status == 'test':
            self.index = test_index
            print("# Test Images len: {}".format(len(self.index)))
            print('Test Ref Index:')
            print(testindex)
        if status == 'val':
            self.index = val_index
            print("# Val Images len : {}".format(len(self.index)))


        self.mos = []
        for line5 in open("./data_copule/mos.txt", "r"):
            line5 = float(line5.strip())
            self.mos.append(line5)
        self.mos = np.array(self.mos)
        #print("mos {}".format (self.mos))
        im_names = []
        ref_names = []
        for line1 in open("./data_copule/im_names.txt", "r"):
            line1 = line1.strip()
            im_names.append(line1)
        im_names = np.array(im_names)
        #print("im_names {}".format (im_names))
        typpe = []
        for line10  in open("./data_copule/im_names.txt", "r"):
            line10 = line10.strip()
            line10 = line10.split("/")
            
            typpe.append(line10)
        typpe =[item[0] for item in typpe]
        typpe = np.array(typpe) 
        
        #print("type {}".format (typpe))

        for line2 in open("./data_copule/refnames.txt", "r"):
            line2 = line2.strip()
            ref_names.append(line2)
        ref_names = np.array(ref_names)
        #print("ref_names {}".format (ref_names))
        copulaFeatures=[]
        for line9 in open("./data_copule/copula_parametre.txt", "r"):
            line9 = float(line9.strip())
            copulaFeatures.append(line9)
        copulaFeatures = np.array(copulaFeatures)
        #print("copulaFeatures {}  ".format (copulaFeatures.shape))
        self.patchesR = ()
        self.patchesL = ()
        self.features = []
        self.label = []
        self.disto = []

        self.im_names = [im_names[i] for i in self.index]
        self.ref_names = [ref_names[i] for i in self.index]
        self.mos = [self.mos[i] for i in self.index]
        typpe = [typpe[i] for i in self.index]
        copulaFeatures=[copulaFeatures[i*108:(i+1)*108] for i in self.index]

        for idx in range(len(self.index)):
            imL = self.gray_loader(Path(str(im_dirL)+"/"+ str(self.im_names[idx])))
            imR = self.gray_loader(Path(str(im_dirR) + "/" + str(self.im_names[idx])))

            patchesR = CropPatches(imR, self.patch_size, self.stride)
            patchesL = CropPatches(imL, self.patch_size, self.stride)

            if status == 'train':
                self.patchesL = self.patchesL + patchesL
                self.patchesR = self.patchesR + patchesR
                
                for i in range(len(patchesL)):
                    self.label.append(self.mos[idx])
                    self.features.append(copulaFeatures[idx])
                    self.disto.append(typpe[idx])

            
            else:
                self.patchesL = self.patchesL + (torch.stack(patchesL), )
                self.patchesR = self.patchesR + (torch.stack(patchesR), )
                self.label.append(self.mos[idx])
                self.features.append(copulaFeatures[idx])
                self.disto.append(typpe[idx])

    def __len__(self):
        return len(self.patchesL)

    def __getitem__(self, idx):
        return self.patchesL[idx],self.patchesR[idx] ,(torch.Tensor([self.label[idx]]), self.features[idx], self.disto[idx])


















