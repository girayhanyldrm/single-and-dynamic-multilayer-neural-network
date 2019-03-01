import numpy as np

class outputLayer:
    def __init__(self,output_size):
        self.a=np.zeros((output_size,1))
        self.z=np.zeros((output_size,1))
        self.count_of_neuron=output_size
        self.prime = np.zeros((output_size,1))
        self.error = np.zeros((output_size,1))
