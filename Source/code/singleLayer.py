import numpy as np
import matplotlib.pyplot as plt
import math
class singleLayer:
    def __init__(self,train_data,train_data_labels,epoch_num,learning_rate,batch_size):
        self.X = self.__normalize(train_data)
        self.labels =self.__modify_labels(train_data_labels)
        self.label=list(train_data_labels[0])
        self.W = np.random.rand(self.X.shape[1] , 5)/100
        self.bias = np.zeros((1,5))
        self.epoch = epoch_num
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_list = []

    def __normalize(self,value):
        return value / 255


    def __modify_labels(self,label):
        new_label = []
        for i in range(0, len(label[0])):
            out = [0, 0, 0, 0, 0]
            index = label[0][i]
            out[index] = 1
            new_label.append(out)
        return new_label

    def __sigmoid(self,x):
        return 1.0 / (1 + np.exp(-x))

    def __softmax (self,w):
        for i in np.arange(0,w.shape[0]):
            w[i]= np.exp(w[i]) /np.sum(np.exp(w[i]))
        return w

    def __batch_file(self):
        batch_of_input = []
        batch_of_label=[]
        bl=[]
        for b in np.arange(0, len(self.X), self.batch_size):
            try:
                batch_of_input.append(self.X[b:b + self.batch_size])
                batch_of_label.append(self.labels[b:b + self.batch_size])
                bl.append(self.label[b:b + self.batch_size])

            except:
                batch_of_input.append(self.X[b:])
                batch_of_label.append(self.labels[b:])
                bl.append(self.label[b:])


        return np.asarray(batch_of_input),np.asarray(batch_of_label),bl

    def __loss(self,probs,bl,size):
        num_examples=size
        correct_logprobs = -np.log(probs[range(num_examples), bl])
        data_loss = np.sum(correct_logprobs) / size
        return data_loss

    def __update_weight(self,X,score):
        dW = np.dot(X.T,score)
        db = np.sum(score, axis=0, keepdims=True)
        self.W += self.learning_rate*dW
        self.bias += self.learning_rate*db


    def __derivative_sigmoid(self,x):
        return x*(1-x)




    def train(self):
        batch_input,batch_label,bl=self.__batch_file()
        for epoch in np.arange(0,self.epoch):
            epoch_loss = []
            for i in np.arange(0,len(batch_input)):
                predict_output =np.dot(batch_input[i],self.W)

                predict_softmax =self.__softmax(predict_output)

                error = batch_label[i] - predict_softmax
                error /= batch_input.shape[0]
                self.__update_weight(batch_input[i],error)

                #loss = self.__loss(predict_softmax,bl,size)
                #epoch_loss.append(loss)

            #self.loss_list.append(sum(epoch_loss))
            #print(sum(epoch_loss))
        np.save('../model/single_layer_model',self.W)

##############################################################################################################
    def train_sigmoid(self):

        batch_input, batch_label, bl = self.__batch_file()
        for epoch in np.arange(0, self.epoch):
            epoch_loss = []
            for i in np.arange(0, batch_input.shape[0]):
                predict_output = self.__sigmoid(np.dot(batch_input[i], self.W))  # 30 x 5 output matris  30x5 30x5


                error =  batch_label[i] - predict_output

                gradient = error*self.__derivative_sigmoid(predict_output)   # learningrate x x.T x (target - output)
                self.W += self.learning_rate * np.dot(batch_input[i].T,gradient)


                loss = np.sum(error ** 2)
                epoch_loss.append(loss)
            print(sum(epoch_loss))
            self.loss_list.append(sum(epoch_loss))

        np.save('single_layer_model',self.W)

        self.plot_loss()

    def predict(self,test_data,model):
        test_data = self.__normalize(test_data)
        pred = self.__sigmoid(test_data.dot(model))
        predict_list = []

        for i in np.arange(0,len(pred)):
            predict_list.append((pred[i].tolist().index(max(pred[i]))))

        return predict_list

    def load_weights(self,weight_numpy):
        self.W = np.load(weight_numpy)

    def plot_loss(self):
        fig = plt.figure()
        plt.plot(np.arange(0, self.epoch), self.loss_list)
        fig.suptitle("Training Loss")
        plt.xlabel("Epoch Number")
        plt.ylabel("Loss Value")
        plt.show()

