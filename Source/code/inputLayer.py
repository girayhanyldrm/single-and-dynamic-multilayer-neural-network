import numpy as np

class inputLayer:
    def __init__(self,input_size,next_layer_neuron_size):
        self.a=np.zeros((input_size,1))
        self.weight = np.random.rand(input_size,next_layer_neuron_size) / 1000
        self.delta = np.zeros(np.shape(self.weight))
