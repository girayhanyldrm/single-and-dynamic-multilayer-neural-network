import numpy as np
class hiddenLayer:
    def __init__(self,neuron_size,next_layer_neuron_size):
        self.z = np.zeros((neuron_size,1))
        self.a = np.zeros((neuron_size,1))
        self.weight = np.random.rand(neuron_size,next_layer_neuron_size) / 100
        self.neuron_size = neuron_size
        self.prime = np.zeros((neuron_size,1))
        self.error = np.zeros((neuron_size,1))
        self.delta = np.zeros(np.shape(self.weight))
