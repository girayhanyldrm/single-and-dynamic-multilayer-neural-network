import argparse
import singleLayer
import scipy.io
import numpy as np


def normalize(value):
    return value / 255

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def predict(test_data,model):
    test_data = normalize(test_data)
    pred = sigmoid(test_data.dot(model))
    predict_list = []

    for i in np.arange(0,len(pred)):
        predict_list.append((pred[i].tolist().index(max(pred[i]))))

    return predict_list
def calculate_accuracy(predict_liste):
    accuracy = 0
    for i in range(0, test_data_label.shape[1]):
        if (predict_liste[i] == test_data_label[0][i]):
            accuracy += 1
    return accuracy*100/test_data_label.shape[1]


parser = argparse.ArgumentParser()
# read the arguments
parser.add_argument('--data_path', '-dp', default='test.mat')
parser.add_argument('--model_path', '-mp',  default='../model/single_layer_model.npy')
args = vars(parser.parse_args())

test_data = scipy.io.loadmat(args['data_path'])
model = np.load(args['model_path'])

test_data_input = test_data["x"]
test_data_label = test_data["y"]

predict=np.asarray(predict(test_data_input,model))
print(calculate_accuracy(predict))