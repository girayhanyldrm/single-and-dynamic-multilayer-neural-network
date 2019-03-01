import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import singleLayer
import multiLayer
import argparse

def visualize(model):
    classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    for i in range(5):
        img = model.T[i][0:768]
        img = np.reshape(img, (32, 24))
        plt.suptitle(classes[i])
        plt.imshow(img, cmap='gray')
        plt.show()


parser = argparse.ArgumentParser()
# read the arguments
parser.add_argument('--data_path', '-dp', default='train.mat')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.02)
parser.add_argument('--batch_size', '-bs', type=int, default=64)
parser.add_argument('--epochs', '-e', type=int, default=100)
parser.add_argument('--liste',type=str,default="10")
args = vars(parser.parse_args())
train_data = scipy.io.loadmat(args['data_path'])
train_data_input = train_data["x"]
train_data_label = train_data["y"]


######### SINGLE LAYER TRAİN AND TEST  #####################################
single_layer = singleLayer.singleLayer(train_data_input,train_data_label,args['epochs'],args['learning_rate'],args['batch_size'])
single_layer.train()

######## MULTI LAYER TRAIN AND TEST  ######################################
hiddensizelist=args['liste'].split(",")
hiddensizelist=[int(x) for x in hiddensizelist]
multi_layer = multiLayer.multiLayer(train_data_input,train_data_label,args['epochs'],args['learning_rate'],args['batch_size'],hiddensizelist)
multi_layer.train()


