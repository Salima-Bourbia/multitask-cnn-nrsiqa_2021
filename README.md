# Multi-Task CNN for Blind SIQA Using Naturalness Analysis

This project is a the code associated to the following paper :

### ** Bourbia Salima, Ayoub Karine, Aladine Chetouani, Mohammed El Hassouni, A Multi-task Convolutional Neural Network For Blind Stereoscopic Image Quality Assessment Using Naturalness Analysis, IEEE International Conference on Image Processing (IEEE - ICIP), ICIP 2021 [http://arxiv.org/abs/2106.09303] **

## Abstract :

This paper addresses the problem of blind stereoscopic image quality assessment (NR-SIQA) using a new multi-task deep learning based-method. In the field of stereoscopic vision, the information is fairly distributed between the left and right views as well as the binocular phenomenon. In this work, we propose to integrate these characteristics to estimate the quality of stereoscopic images without reference through a convolutional neural network. Our method is based on two main tasks: the first task predicts naturalness analysis based features adapted to stereo images, while the second task predicts the quality of such images. The former, so-called auxiliary task, aims to find more robust and relevant features to improve the quality prediction. To do this, we compute naturalness-based features using a Natural Scene Statistics (NSS) model in the complex wavelet domain. It allows to capture the statistical dependency between pairs of the stereoscopic images. 

 ## Architecture of the model :

![alt text](https://github.com/salima000/CopulaCNN/blob/main/network.PNG)



## Virtual environment link :

      https://drive.google.com/file/d/1mCqKukigd_ag52qKK55gMwQA3Ax6mOWe/view?usp=sharing

                      
                        
## To activate the virtual environment :
   
      source ./copule/bin/activate



## Dependencies : 

      
      python version 3.6
      
      torch  0.4
      
      torchvision 0.2
      
      tensorboard

      Pillow

      numpy

      opencv

      scipy

      PyYAML
      
      scikit-image
      
      
 
 
 Note: using a Linux distribution such as Ubuntu is highly recommended     
      
## To launch the code :
        
        python train.py

## Visualization : 
 
        tensorboard --logdir=visualize/tensorboard


## To exit the virtual environment :
      
        deactivate
   

## Citation :

If you found this code useful,  we would be grateful if you cite the paper :


     @inproceedings{bourbia:hal-03258262,
       TITLE = {{A Multi-task convolutional neural network for blind stereoscopic image quality assessment using naturalness analysis}},
       AUTHOR = {Bourbia, Salima and Karine, Ayoub and Chetouani, Aladine and El Hassouni, Mohammed},
       URL = {https://hal.archives-ouvertes.fr/hal-03258262},
       BOOKTITLE = {{The 28th IEEE International Conference on Image Processing (IEEE - ICIP)}},
       ADDRESS = {Anchorage-Alaska, France},
       YEAR = {2021},
       MONTH = Sep,
       HAL_ID = {hal-03258262},
       HAL_VERSION = {v1},
     }

