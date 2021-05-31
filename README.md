# CopulaCNN


PyTorch 0.4.0 implementation of the following paper:
Bourbia Salima, Ayoub Karine, Aladine Chetouani, Mohammed El Hassouni, 
A MULTI-TASK CONVOLUTIONAL NEURAL NETWORK FOR BLIND STEREOSCOPICIMAGE QUALITY ASSESSMENT USING NATURALNESS ANALYSIS// LINK



# data base Link : 
        

        https://drive.google.com/file/d/1A-DalUofuwYHJn3jGeM3ht8FCwCoGlhy/view?usp=sharing
        

# Virtual environment link :

                        https://drive.google.com/file/d/1mCqKukigd_ag52qKK55gMwQA3Ax6mOWe/view?usp=sharing
                        
                        
# Activate the virtual environment :
   
      source ./copule/bin/activate



# Dependencies :

      
      python version 3.6.9
      
      pip install torch==0.4.1

      pip install 'git+https://github.com/lanpa/tensorboardX'

      pip install Pillow==6.2.1

      pip install numpy==1.17.3

      pip install opencv-python

      pip install scipy==1.3.1

      pip install torchvision==0.2.2
     
      pip install PyYAML
      
      python -m pip install -U scikit-image
      
      pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
      
      
# To launch the code, you just have to launch the file "python train.py

after running the code, the number of epochs will continue to be displayed in the cmd to keep you informed of the progress of the execution. in our code, we used 500 epochs with 586 
in our code we used 500 epochs with 586 iterations each.

# To leave the virtual environment :
      
        deactivate
   
   
 # Architecture of the model:

![alt text](https://github.com/br-salima/Deep_cnn_HVS/blob/master/archi-dual.PNG?raw=true)


