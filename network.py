import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchsummary import summary
#COPULACNN

class COPULACNN(nn.Module):
    def __init__(self):
        super(COPULACNN, self).__init__()
        # conv of left view
        self.conv1L = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2L = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3L = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4L = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5L = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        # conv of right view
        self.conv1R = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2R = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3R = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4R = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5R = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        #conv sub network D 
        self.conv4C = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5C = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)

        # FC of letf view
        self.fc1_L = nn.Linear(2048, 512)
        self.fc2_L = nn.Linear(512, 512)
        # FC of right view
        self.fc1_R = nn.Linear(2048, 512)
        self.fc2_R = nn.Linear(512, 512)

        # FC of subnetwork D
        self.fc1_bas = nn.Linear(2048, 512)
        self.fc2_bas = nn.Linear(512, 512)
        # FC of subnetwork C
        self.fc1_haut = nn.Linear(8192, 512)
        self.fc2_haut = nn.Linear(512, 512)
        # FC of task 1 
        self.fc1_1 = nn.Linear(2048, 1024)
        self.fc2_1 = nn.Linear(1024, 108)
        # FC of task 2 
        self.fc1_2 = nn.Linear(2048, 1024)
        self.fc2_2 = nn.Linear(2048, 1024)
        self.fc3_2 = nn.Linear(1024, 1)

        
    def forward(self, xL,xR):

        x_distort_L = xL.view(-1, xL.size(-3), xL.size(-2), xL.size(-1))
        x_distort_R = xR.view(-1, xR.size(-3), xR.size(-2), xR.size(-1))

     ####################################################### left view  ####################################################  
        
        x1L = F.relu(self.conv1L(x_distort_L)) 
        x1L = F.max_pool2d(x1L, (2, 2), stride=2)#32x16×16

        x2L = F.relu(self.conv2L(x1L))#32x8×8
        x2L = F.max_pool2d(x2L, (2, 2), stride=2)

        x3L = F.relu(self.conv3L(x2L))#64x8×8

        x4L = F.relu(self.conv4L(x3L))#64x8×8

        x5L = F.relu(self.conv5L(x4L))
        x5L = F.max_pool2d(x5L, (2, 2), stride=2)#128x4×4

        fcL = x5L.view(-1, self.num_flat_features(x5L))
        fc1L = self.fc1_L(fcL)#512
        fc1L = F.dropout(fc1L, p=0.35, training=True, inplace=False)

        fc2L = self.fc2_L(fc1L)#512
        fc2L = F.dropout(fc2L, p=0.5, training=True, inplace=False)
 
                    

     ####################################################### right view  ####################################################  
        
        
        x1R = F.relu(self.conv1R(x_distort_R))
        x1R = F.max_pool2d(x1R, (2, 2), stride=2)

        x2R = F.relu(self.conv2R(x1R))
        x2R =F.max_pool2d(x2R, (2, 2), stride=2)
      
        x3R = F.relu(self.conv3R(x2R))
        x4R = F.relu(self.conv4R(x3R))
        

        x5R = F.relu(self.conv5R(x4R))
        x5R = F.max_pool2d(x5R, (2, 2), stride=2)
        
        fcR = x5R.view(-1, self.num_flat_features(x5R))
        fc1R = self.fc1_R(fcR)
        fc1R = F.dropout(fc1R, p=0.35, training=True, inplace=False)

        fc2R = self.fc2_R(fc1R)
        fc2R = F.dropout(fc2R, p=0.5, training=True, inplace=False)
        
        ###################################################### subnetwork D #############################################
        add =torch.add(x2L, x2R)
        diff = torch.sub(x2L, x2R)
        cat1 = F.relu(torch.cat((add, diff), 1))

        A4 = F.relu(self.conv4C(cat1))
        A5 = F.relu(self.conv5C(A4))
        A5 = F.max_pool2d(A5, (2, 2), stride=2)

     
        fc_cat1 = A5.view(-1, self.num_flat_features(A5))
        fc1_L_cat1 = self.fc1_bas(fc_cat1)
        fc1_L_cat1 = F.dropout(fc1_L_cat1, p=0.35, training=True, inplace=False)

        fc2_cat1= self.fc2_bas(fc1_L_cat1)
        fc2_cat1 = F.dropout(fc2_cat1, p=0.5, training=True, inplace=False)


        #############################################   subnetwork D  ############################

        add_con2 =torch.add(x4L, x4R)
        diff_con2 = torch.sub(x4L, x4R)
        cat2 = F.relu(torch.cat((add_con2, diff_con2), 1))
        
        
        L_cat2 = cat2.view(-1, self.num_flat_features(cat2))
        L = self.fc1_haut(L_cat2)
        dr = F.dropout(L, p=0.35, training=True, inplace=False)

        f2= self.fc2_haut(dr)
        f2 = F.dropout(f2, p=0.5, training=True, inplace=False)

        #############################################    Concatenation  ############################

        cat_total = F.relu(torch.cat((fc2L, fc2R,fc2_cat1,f2), 1))
        fc_total = cat_total.view(-1, self.num_flat_features(cat_total))

        ################################## tasks prediction  ##########################################


        fc1_1 = self.fc1_1(fc_total)
        fc2_1 = self.fc2_1(fc1_1)  # Naturalness based-features prediction 


        fc1_2 = self.fc1_2(fc_total)
        cat = F.relu(torch.cat((fc1_2, fc1_1), 1))
        fc2_2 = self.fc2_2(cat)
        quality = self.fc3_2(fc2_2) #quality score prediction

        return quality, fc2_1


    def num_flat_features(self, xx):
        size = xx.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features












