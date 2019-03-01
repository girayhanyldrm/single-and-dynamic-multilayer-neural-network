import numpy as np
import matplotlib.pyplot as plt
import inputLayer
import hiddenLayer
import outputLayer


class multiLayer:
    def __init__(self,train_data,train_data_labels,epoch_num,learning_rate,batch_size,hidden_layer_list):
        self.X = self.__normalize(train_data)
        self.labels =self.__modify_labels(train_data_labels)
        self.label=list(train_data_labels[0])
        self.W = np.random.rand(self.X.shape[1] , 5)/1000
        self.epoch = epoch_num
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_list = []
        self.neural_network=[]
        self.hidden_layer_list=hidden_layer_list


        self.neural_network.append(self.create_input_layer(self.hidden_layer_list[0]))
        for i in range(0,len(self.hidden_layer_list)-1):
            self.neural_network.append(self.create_hidden_layer(self.hidden_layer_list[i],self.hidden_layer_list[i+1]))
        self.neural_network.append(self.create_hidden_layer(self.hidden_layer_list[-1], 5))
        self.neural_network.append(self.create_output_layer())

######      CREATİNG OBJECT FUNCTİONS           #############
    def create_input_layer(self, next_layer_neuron_size):
        input_layer = inputLayer.inputLayer(self.X.shape[1], next_layer_neuron_size)
        return input_layer

    def create_hidden_layer(self,neuron_size,next_layer_neuron_size):
        hidden_layer = hiddenLayer.hiddenLayer(neuron_size, next_layer_neuron_size)
        return hidden_layer

    def create_output_layer(self):
        output_layer = outputLayer.outputLayer(5)
        return output_layer

#####       PRIVATE CLASS FUNCTIONS             #############
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

    def __sigmoid(self, s):
        # activation function
        return 1 / (1 + np.exp(-s))

    def __sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def __tan(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    def __tan_prime(self, a):
        return 1 - np.square(a)

    def __reLU(self, x):
        return np.maximum(x, 0)

    def __reLU_prime(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

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

    def __loss(self,true):
        out_layer = self.neural_network[-1]
        out_layer.error = (true - out_layer.a) * self.__sigmoidPrime(out_layer.a)
        return np.sum(self.neural_network[-1].error ** 2)

    def __cross_entropy(self,probs,bl,size):
        num_examples=size
        correct_logprobs = -np.log(probs[range(num_examples), bl])
        cross_entropy = np.sum(correct_logprobs) / size
        return cross_entropy



    def train(self):
        batch_input,batch_label,bl=self.__batch_file()
        for epoch in np.arange(0,self.epoch):
            epoch_list=[]
            for i in np.arange(0,len(batch_input)):

                self.neural_network[0].a = batch_input[i]   #30 x 5
                self.forward()

                self.neural_network[-1].error = batch_label[i] - self.neural_network[-1].a

                epoch_list.append(np.sum(self.neural_network[-1].error ** 2))
                self.backward()

            #print(sum(epoch_list))
            self.loss_list.append(sum(epoch_list))
        #self.plot_loss()
        for i in range(0,len(self.neural_network)-1):
            liste=[]
            liste.append(self.neural_network[i].weight)
        np.save('../model/multi_layer_model',liste)

    def forward(self):
        for i in np.arange(0,len(self.neural_network)-1):
            self.neural_network[i+1].z = np.dot(self.neural_network[i].a,self.neural_network[i].weight)
            self.neural_network[i+1].a = self.__sigmoid(self.neural_network[i+1].z)
            self.neural_network[i+1].prime = self.__sigmoidPrime(self.neural_network[i+1].a)

    def backward(self):
        for i in np.arange((len(self.neural_network)-1) , 1 , -1):
            self.neural_network[i].delta = (self.neural_network[i].error * self.neural_network[i].prime)
            self.neural_network[i-1].error = np.dot(self.neural_network[i].delta,self.neural_network[i-1].weight.T)     # 30 x 5  . 5 x 4    30 x 4
            self.neural_network[i-1].weight += self.learning_rate * np.dot(self.neural_network[i-1].z.T,self.neural_network[i].delta) # 30 x 4 30 x 5

        self.neural_network[1].delta = (self.neural_network[1].error * self.neural_network[1].prime)
        self.neural_network[0].weight += self.learning_rate * np.dot(self.neural_network[0].a.T,self.neural_network[1].delta)

    def predict(self,test_data):

        test_data = self.__normalize(test_data)
        pred = self.test_forward(test_data)
        predict_list = []
        for i in np.arange(0,len(pred)):
            predict_list.append((pred[i].tolist().index(max(pred[i]))))

        return predict_list

    def test_forward(self,test):
        self.neural_network[0].a=test
        for i in np.arange(0,len(self.neural_network)-1):
            self.neural_network[i+1].z = np.dot(self.neural_network[i].a,self.neural_network[i].weight)
            self.neural_network[i+1].a = self.__sigmoid(self.neural_network[i+1].z)
        return self.neural_network[i+1].a

    def plot_loss(self):
        fig = plt.figure()
        plt.plot(np.arange(0, self.epoch), self.loss_list)
        fig.suptitle("Training Loss")
        plt.xlabel("Epoch Number")
        plt.ylabel("Loss Value")
        plt.show()