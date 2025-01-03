import numpy as np
import random

class ReplyBuffer():
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.buffer = []
        self.size = 0
        self.max_length = 320
        pass

    def add(self, x):
        if len(self.buffer) < self.max_length:
            self.buffer.append(x)
        else:
            self.buffer = self.buffer[self.max_length:]
            self.buffer.append(x)
        pass

    def sample(self, ):

        if self.size < self.batch_size:
            B = self.buffer[:]
            #return self.buffer
            random.shuffle(B)
            return B
        else:
            B = self.buffer[self.size - self.batch_size:]
            #return self.buffer[self.size - self.batch_size:]
            random.shuffle(B)
            return B
        pass
    
    def shuffle(self, ):
        pass

    def clear(self, ):
        self.buffer = []
        self.size = 0
        pass

