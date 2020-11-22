import numpy as np
import random
import keras
from scipy.spatial import distance

class Generator:
    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.unique_labels = list(set(self.y))

        # Create a dictionary with the labels as the keys
        self.label_to_data = {label: [] for label in self.unique_labels}
        print("unique_labels:",len(self.unique_labels))
        for i, label in enumerate(self.y):
            self.label_to_data[label].append(i)


    def get_data_item(self, label):
        """ Choose an data point from X with the given label """
        idx = random.choice(self.label_to_data[label])
        data = self.X[idx]
        return data, idx

    
    def get_triplet(self):
        """
        Choose a triplet (anchor, positive, negative) of data points from X such that anchor
        and positive have the same label, anchor and negative have different labels
        """
        n_label = a_label = np.random.choice(self.unique_labels)
        while n_label == a_label:
            ## keep searching randomly
            n_label = np.random.choice(self.unique_labels)
        anchor, i1 = self.get_data_item(a_label)
        positive, i2 = self.get_data_item(a_label)

        # Make sure that anchor and positive are not the same data points
        loopout = 0
        while i1 == i2:
            if loopout == 10:
                break
            anchor, i1 = self.get_data_item(a_label)
            positive, i2 = self.get_data_item(a_label)
            loopout += 1  
#         if i1 == i2:
#             print("duplicate!", a_label)
        negative, _ = self.get_data_item(n_label)
        return anchor, positive, negative

    
    def generate_data(self, batch_size):
        """Generate an un-ending stream (ie a generator) of data for
        training or test."""
        while True:
            data = []
            labels = []

            for i in range(batch_size):
                anchor, positive, negative = self.get_triplet()
                if all(anchor == positive) or anchor is positive:
                    i -= 1
                    continue
                
                data.append(np.concatenate((anchor, positive, np.array([distance.euclidean(anchor, positive),
                                                                       distance.cosine(anchor, positive)]))))
                labels.append(1)
                data.append(np.concatenate((anchor, negative, np.array([distance.euclidean(anchor, negative),
                                                                       distance.cosine(anchor, negative)]))))
                labels.append(0)
            data = np.array(data, dtype='float32')
            labels = np.array(labels, dtype='float32')    
            labels = keras.utils.to_categorical(labels, 2)    
            yield data, labels
