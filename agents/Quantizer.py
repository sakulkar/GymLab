from __future__ import division # for floating point division
from math import floor

class Quantizer(object):
    """Observation Qauntizer
    This class is used for quantizing the observations into discrete states to be used for QTable.QAgent
    """
    def __init__(self, low, high, buckets):
        """
        Parameters
        ----------
        low : List/tuple of Lowest possisble observation values
        high : List/tuple of Highest possisble observation values
        buckets : Number of buckets to quantize the dimension into (List or tuple)
        kwargs : Extra arguments passed (Not needed)
        -------
        """
        # static attributes
        self.low = low          # Lowest list of Observations
        self.high = high        # Highest list of Observations
        self.buckets = buckets  # Number of buckets per dimension
        self.dim = len(low)     # Dimension of the observation
        print self.dim

        self.width = []        # width of each quantization step
        for idx in range(self.dim):
            self.width.append((self.high[idx] - self.low[idx]) / self.buckets[idx])            

    def quantize(self, observation):
        """Quantize the observation
        """
        quantized_obs = []
        for idx in range(self.dim):
            if observation[idx] < self.low[idx]:
                quantized_obs.append(0)
            elif observation[idx] >= self.high[idx]:
                quantized_obs.append(self.buckets[idx]-1)
            else:
                quantized_obs.append(int(floor((observation[idx] - self.low[idx])/ self.width[idx])))

        return tuple(quantized_obs)
