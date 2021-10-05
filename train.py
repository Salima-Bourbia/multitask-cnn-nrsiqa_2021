"""
PyTorch 0.4.0 implementation of the following paper:
Bourbia Salima, Ayoub Karine, Aladine Chetouani, Mohammed El Hassouni, 
A MULTI-TASK CONVOLUTIONAL NEURAL NETWORK FOR BLIND STEREOSCOPICIMAGE QUALITY ASSESSMENT USING NATURALNESS ANALYSIS//

"""

import torch
import os
import numpy as np
from scipy import stats
import yaml
from argparse import ArgumentParser
import random
import torch.nn.functional as F
import torch.nn as nn
import sys
import torchvision
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

from network import COPULACNN
from SIQADataset import SIQADataset



def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_indexNum(config, index, status):
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
        train_index.append(i) if (ref_ids[i] in trainindex) else \
            test_index.append(i) if (ref_ids[i] in testindex) else \
                val_index.append(i)
    if status == 'train':
        index = train_index
    if status == 'test':
        index = test_index
    if status == 'val':
        index = val_index

    return len(index)


if __name__ == '__main__':
    # Training settings
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dataset", type=str, default="LIVE_Phase1")
    parser.add_argument("--weight_decay", type=float, default=0.0001)

    args = parser.parse_args()
    
    seed = random.randint(10000000, 99999999)
    torch.manual_seed(seed)
    np.random.seed(seed)
    print("seed:", seed)

    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    index = []
    if args.dataset == "LIVE_Phase1":
        print("dataset: LIVE_Phase1")
        index = list(range(1, 21))
        random.shuffle(index)
    print('rando index', index)
    



    ensure_dir('results')
    save_model = "./results/model.pth" 
    ensure_dir('visualize/tensorboard')
    writer = SummaryWriter('visualize/tensorboard')


    dataset = args.dataset
    valnum = get_indexNum(config, index, "val")
    testnum = get_indexNum(config, index, "test")

    train_dataset = SIQADataset(dataset, config, index, "train")
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0)


    val_dataset = SIQADataset(dataset, config, index, "val")
    val_loader = torch.utils.data.DataLoader(val_dataset)
    test_dataset = SIQADataset(dataset, config, index, "test")
    test_loader = torch.utils.data.DataLoader(test_dataset)



    ###model
    model = COPULACNN().to(device)
    print(model)
    ###
    criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    torch.optim.lr_scheduler.StepLR(optimizer, 200, gamma=0.1, last_epoch=-1)
    ###
    best_SROCC = -1
    # training phase
    for epoch in range(args.epochs):
        
        model.train()
        LOSS_all = 0
        LOSS_copula = 0
        LOSS_q = 0
        for i, (patchesL, patchesR,(label, features,disto)) in enumerate(train_loader):
            patchesL = patchesL.to(device)
            patchesR = patchesR.to(device)
            label = label.to(device)
            features = features.to(device).float()
            

            optimizer.zero_grad()
            outputs_q = model(patchesL,patchesR)[0]
            outputs_copula = model(patchesL,patchesR)[1]
            loss_copula = criterion(outputs_copula, features)

            #print('loss_copula',loss_copula)
            loss_q = criterion(outputs_q, label)
            #print('loss_q',loss_q)
            loss = 25*loss_copula + loss_q
            #print ('loss',loss)
            loss.backward()
            optimizer.step()

            LOSS_all = LOSS_all + loss.item()
            LOSS_copula = LOSS_copula + loss_copula.item()
            LOSS_q = LOSS_q + loss_q.item()
          
        train_loss_all = LOSS_all / (i + 1)
        print ('training_loss',train_loss_all)
        train_loss_copula = LOSS_copula / (i + 1)
        print('copula_training_loss',train_loss_copula)
        train_loss_q = LOSS_q / (i + 1)
        print('quality score training loss',train_loss_q)

        writer.add_scalar('quality score loss', train_loss_q,  epoch * len(train_loader) + i)
        writer.add_scalar('copula features prediction loss',train_loss_copula,  epoch * len(train_loader) + i)
        writer.add_scalar('total_loss',train_loss_all, epoch * len(train_loader) + i)
        

        # validation phase
        y_predval = np.zeros(valnum)
        y_val = np.zeros(valnum)  

        model.eval()
        L = 0
   
        with torch.no_grad():   
            for i, (patchesL,patchesR, (label, features, disto)) in enumerate(val_loader):

                patchesL = patchesL.to(device)
                patchesR = patchesR.to(device)
                label = label.to(device)

                y_val[i] = label.item()

                disto = disto[0]

                outputs = model(patchesL,patchesR)[0]

                score = outputs.mean()
                y_predval[i] = score

                loss = criterion(score, label[0])
                L = L + loss.item()

        val_loss = L / (i + 1)
        val_SROCC = stats.spearmanr(y_predval, y_val)[0]
        val_PLCC = stats.pearsonr(y_predval, y_val)[0]
        val_KROCC = stats.stats.kendalltau(y_predval, y_val)[0]
        val_RMSE = np.sqrt(((y_predval - y_val) ** 2).mean())



        writer.add_scalar("validation/val_SROCC", val_SROCC,  epoch * len(val_loader) + i)
        writer.add_scalar("validation/val_PLCC", val_PLCC,epoch * len(val_loader) + i)
        writer.add_scalar("validation/val_KROCC", val_KROCC, epoch * len(val_loader) + i)
        writer.add_scalar("validation/val_RMSE", val_RMSE,epoch * len(val_loader) + i)

        # test phase
        y_pred = np.zeros(testnum)
        y_test = np.zeros(testnum)
        L = 0

        with torch.no_grad():
            for i, (patchesL,patchesR, (label, features,disto)) in enumerate(test_loader):
                patchesL = patchesL.to(device)
                patchesR = patchesR.to(device)
                label = label.to(device)

                y_test[i] = label.item()
 


                disto = disto[0]

                outputs = model(patchesL,patchesR)[0]
                score = outputs.mean()
                y_pred[i] = score

                loss = criterion(score, label[0])
                L = L + loss.item()

        test_loss = L / (i + 1)
        SROCC = stats.spearmanr(y_pred, y_test)[0]
        PLCC = stats.pearsonr(y_pred, y_test)[0]
        KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
        RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())

        writer.add_scalar("test/test_SROCC", SROCC,  epoch * len(test_loader) + i)
        writer.add_scalar("test/test_PLCC", PLCC,epoch * len(test_loader) + i)
        writer.add_scalar("test/test_KROCC", KROCC, epoch * len(test_loader) + i)
        writer.add_scalar("test/test_RMSE", RMSE,epoch * len(test_loader) + i)
    
        print("Epoch {} Valid Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(epoch,
                                                                                                             val_loss,
                                                                                                             val_SROCC,
                                                                                                             val_PLCC,
                                                                                                             val_KROCC,
                                                                                                             val_RMSE))
        print("Epoch {} Test Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(epoch,
                                                                                                            test_loss,
                                                                                                            SROCC,
                                                                                                            PLCC,
                                                                                                            KROCC,
                                                                                                            RMSE))





        if val_SROCC > best_SROCC :
            print("Update Epoch {} best valid SROCC".format(epoch))
            print("Valid Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(val_loss,
                                                                                                                val_SROCC,
                                                                                                                val_PLCC,
                                                                                                                val_KROCC,
                                                                                                                val_RMSE))
            print("Test Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(test_loss,
                                                                                                            SROCC,
                                                                                                            PLCC,
                                                                                                            KROCC,
                                                                                                RMSE))

            torch.save(model.state_dict(), save_model)
            best_SROCC = val_SROCC
    writer.close()
    ########################################################################## final test ############################################
    model.load_state_dict(torch.load(save_model))
    model.eval()
    with torch.no_grad():
        y_pred = np.zeros(testnum)
        y_test = np.zeros(testnum)

        ###################### copula

        y_pred_cp = np.zeros((testnum,108))
        y_test_cp = np.zeros((testnum,108))
        P= np.zeros((1,108))


        ######################### disto types
        y_pred1 = []
        y_test1 =[]
        y_pred2 = []
        y_test2 =[]
        y_pred3 = []
        y_test3 =[]
        y_pred4 = []
        y_test4 =[]
        y_pred5 = []
        y_test5 =[]

        L = 0
    
        for i, (patchesL,patchesR, (label, features,disto)) in enumerate(test_loader):
 
            patchesL = patchesL.to(device)
            patchesR = patchesR.to(device)
            label = label.to(device)

            y_test[i] = label.item()
            y_test_cp[i]  = features
            features = features.to(device).float()
            disto = disto[0]

            outputs = model(patchesL,patchesR)[0]
            cp = model(patchesL,patchesR)[1]

            M= 0
            for j in range(cp.shape[1]) : 

                for k in range(cp.shape[0]) : 
                    M= cp[k,j] +M
                   
                P[0,j]= M/(k+1) 

                M= 0 

            y_pred_cp[i,:] = P

            score = outputs.mean()
            y_pred[i] = score

            if disto == 'jp2k' :
                
                y_pred1.append(y_pred[i])
                y_test1.append(y_test[i])


            if disto == 'jpeg' :

                y_pred2.append(y_pred[i])
                y_test2.append(y_test[i])


            if disto == 'wn' :

                y_pred3.append(y_pred[i])
                y_test3.append(y_test[i])

            if disto == 'gblur' :

                y_pred4.append(y_pred[i])
                y_test4.append(y_test[i])

            if disto == 'fastfading' :

                y_pred5.append(y_pred[i])
                y_test5.append(y_test[i])


            loss = criterion(score, label[0])
            L = L + loss.item()

    test_loss = L / (i + 1)

    y_pred1  = np.array(y_pred1)
    y_test1  = np.array(y_test1)

    y_pred2  = np.array(y_pred2)
    y_test2  = np.array(y_test2)

    y_pred3  = np.array(y_pred3)
    y_test3  = np.array(y_test3)

    y_pred4  = np.array(y_pred4)
    y_test4  = np.array(y_test4)

    y_pred5  = np.array(y_pred5)
    y_test5  = np.array(y_test5)

    #################################################### SROCC/PLCC/KROCC/RMSE score
    SROCC = stats.spearmanr(y_pred, y_test)[0]
    PLCC = stats.pearsonr(y_pred, y_test)[0]
    KROCC = stats.stats.kendalltau(y_pred, y_test)[0]
    RMSE = np.sqrt(((y_pred - y_test) ** 2).mean())

    ################################## SROCC/PLCC/KROCC/RMSE copula

    RMSE_cp = np.sqrt(((y_pred_cp - y_test_cp) ** 2).mean())


    ####################### SROCC/PLCC/KROCC/RMSE score for each disto type

    SROCCtst1 = stats.spearmanr(y_pred1, y_test1)[0]
    PLCCtst1 = stats.pearsonr(y_pred1, y_test1)[0]
    KROCCtst1 = stats.stats.kendalltau(y_pred1, y_test1)[0]
    RMSEtst1 = np.sqrt(((y_pred1 - y_test1) ** 2).mean())

        

    SROCCtst2 = stats.spearmanr(y_pred2, y_test2)[0]
    PLCCtst2 = stats.pearsonr(y_pred2, y_test2)[0]
    KROCCtst2 = stats.stats.kendalltau(y_pred2, y_test2)[0]
    RMSEtst2 = np.sqrt(((y_pred2 - y_test2) ** 2).mean())


    SROCCtst3 = stats.spearmanr(y_pred3, y_test3)[0]
    PLCCtst3 = stats.pearsonr(y_pred3, y_test3)[0]
    KROCCtst3 = stats.stats.kendalltau(y_pred3, y_test3)[0]
    RMSEtst3 = np.sqrt(((y_pred3 - y_test3) ** 2).mean())


    SROCCtst4 = stats.spearmanr(y_pred4, y_test4)[0]
    PLCCtst4 = stats.pearsonr(y_pred4, y_test4)[0] 
    KROCCtst4 = stats.stats.kendalltau(y_pred4, y_test4)[0]
    RMSEtst4 = np.sqrt(((y_pred4 - y_test4) ** 2).mean())


    SROCCtst5 = stats.spearmanr(y_pred5, y_test5)[0]
    PLCCtst5 = stats.pearsonr(y_pred5, y_test5)[0]
    KROCCtst5 = stats.stats.kendalltau(y_pred5, y_test5)[0]
    RMSEtst5 = np.sqrt(((y_pred5 - y_test5) ** 2).mean())
 

    if os.path.exists('total_result.txt'):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not
    with open('total_result.txt', 'a+') as f:
        f.seek(1)
        f.write("%s\n" % "Phase 1 : Final test Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(test_loss,
                                                                                                                SROCC,
                                                                                                                PLCC,
                                                                                                                KROCC,
                                                                                                                RMSE))
        print("Phase 1 : Final test Results: loss={:.3f} SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(test_loss,
                                                                                                                SROCC,
                                                                                                                PLCC,
                                                                                                                KROCC,
                                                                                                                RMSE))


        f.close() 
    if os.path.exists('type_disto_result.txt'):
            append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not
    with open('type_disto_result.txt', 'a+') as f:
        f.seek(1)
                                                                                                                                                                                                                  
        print("Final test phase 1 - jp2k distortion Results: SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(SROCCtst1,
                                                                                                                PLCCtst1,
                                                                                                                KROCCtst1,
                                                                                                                RMSEtst1))
        f.write("%s\n" % "Final test phase 1 - jp2k distortion Results: SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(
                                                                                                                SROCCtst1,
                                                                                                                PLCCtst1,
                                                                                                                KROCCtst1,
                                                                                                                RMSEtst1))

        print("Final test  phase 1 - jpeg distortion Results: SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(
                                                                                                                    
                                                                                                                        SROCCtst2,
                                                                                                                        PLCCtst2,
                                                                                                                        KROCCtst2,
                                                                                                                        RMSEtst2))
        f.write("%s\n" % "Final test  phase 1 - jpeg distortion Results: SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(
                                                                                                                    
                                                                                                                        SROCCtst2,
                                                                                                                        PLCCtst2,
                                                                                                                        KROCCtst2,
                                                                                                                        RMSEtst2))

        print("Final test  phase 1 - WN  distortion Results: SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(
                                                                                                                    
                                                                                                                        SROCCtst3,
                                                                                                                        PLCCtst3,
                                                                                                                        KROCCtst3,
                                                                                                                        RMSEtst3))

        f.write("%s\n" %"Final test  phase 1 - WN distortion Results: SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(
                                                                                                                    
                                                                                                                        SROCCtst3,
                                                                                                                        PLCCtst3,
                                                                                                                        KROCCtst3,
                                                                                                                        RMSEtst3))



        print("Final test  phase 1 - gblur distortion Results: SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(
                                                                                                                    
                                                                                                                        SROCCtst4,
                                                                                                                        PLCCtst4,
                                                                                                                        KROCCtst4,
                                                                                                                        RMSEtst4))
        f.write("%s\n" %"Final test  phase 1 - gblur distortion Results: SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(
                                                                                                                    
                                                                                                                        SROCCtst4,
                                                                                                                        PLCCtst4,
                                                                                                                        KROCCtst4,
                                                                                                                        RMSEtst4))


        print("Final test  phase 1 - fastfading distortion Results: SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(
                                                                                                                    
                                                                                                                        SROCCtst5,
                                                                                                                        PLCCtst5,
                                                                                                                        KROCCtst5,
                                                                                                                        RMSEtst5))

        f.write("%s\n" %"Final test  phase 1 - fastfading distortion Results: SROCC={:.3f} PLCC={:.3f} KROCC={:.3f} RMSE={:.3f}".format(
                                                                                                                    
                                                                                                                        SROCCtst5,
                                                                                                                        PLCCtst5,
                                                                                                                        KROCCtst5,
                                                                                                                        RMSEtst5))
        f.close() 

    if os.path.exists('copula_predection_result.txt'):
        append_write = 'a' # append if already exists
    else:
        apend_write = 'w' # make a new file if not
    with open('copula_predection_result.txt', 'a+') as f:
        f.seek(1)


        print("Final test copula prediction Results: SROCC={:.3f} ".format( RMSE_cp))

        f.write("%s\n" % "Final test copula prediction Results: SROCC={:.3f} ".format( RMSE_cp))


        f.close() 





