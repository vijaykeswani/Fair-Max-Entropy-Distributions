"""
Class for domain object
"""


import numpy as np


class Domain(object):
    def __init__(self, labels, uniqueValues):
        """
        :param labels: sequence of strings, used for retrieving features
        :param uniqueValues: list of unique values for each feature
                order of unique values should match to labels
                unique value of individual features should be list of distinct values
                distinct values can be either float, list of same size or numpy vectors of same size
        """
        self.labels = labels
        self.uniqueValues = [np.array(uv, dtype=np.float) for uv in uniqueValues]
        for i, uv in enumerate(self.uniqueValues):
            if len(uv.shape) == 1:
                self.uniqueValues[i] = uv.reshape(-1, 1)

    def getSizes(self):
        return [uniqueValues.shape[0] for uniqueValues in self.uniqueValues]

    @property
    def size(self):
        """
        :return: number of unique points in the domain
        """
        result = 1
        for uniqueValues in self.uniqueValues:
            result *= uniqueValues.shape[0]
        return result

    @property
    def numberOfFeatures(self):
        """
        ;return; number of features in the domain (length of labels)
        """
        return len(self.labels)

    def getUniqueValues(self, label, featureIndex=None):
        """
        :param label: label of the feature
        ;param featureIndex; index of the feature, if it is provided,
                    then label is ignored, otherwise computed from label
        :return: corresponding unique values
        """
        if featureIndex is None:
            featureIndex = self.labels.index(label)
        return self.uniqueValues[featureIndex]

    def dimensionOfFeature(self, label, featureIndex=None):
        """
        :param label: label of the feature
        ;param featureIndex; index of the feature, if it is provided,
                    then label is ignored, otherwise computed from label
        :return:
        """
        uniqueValues = self.getUniqueValues(label, featureIndex)
        try:
            return len(uniqueValues[0])
        except TypeError:
            # feature is numerical
            return 1

    @property
    def dimension(self):
        """
        :return: dimension of domain after each feature expanded
        """
        return sum([self.dimensionOfFeature(None, i) for i in range(self.numberOfFeatures)])

    def __str__(self):
        return "Domain in {dimension} with {features}".format(
                                                dimension=self.dimension,
                                                features=self.numberOfFeatures)

    def __repr__(self):
        return str(self)

    def toIndicesFeature(self, label, data, featureIndex=None):
        """
        :param label:
        :param data:
        :param featureIndex:
        :return:
        """
        uniqueValues = self.getUniqueValues(label, featureIndex)
        newTypeSize = uniqueValues.dtype.itemsize * uniqueValues.shape[1]
        newType = np.dtype((np.void, newTypeSize))
        uniqueValues = np.ascontiguousarray(uniqueValues).view(newType).flatten()
        data = np.ascontiguousarray(data).view(newType).flatten()
        mapping = {value.tobytes(): index for index, value in enumerate(uniqueValues)}
        return [mapping[row.tobytes()] for row in data]

    def fromIndicesFeature(self, label, data, featureIndex=None):
        """
        :param label:
        :param data:
        :param featureIndex:
        :return:
        """
        uniqueValues = self.getUniqueValues(label, featureIndex)
        return uniqueValues[data]


    def toIndices(self, data):
        """
            converts given matrix consisting of points from the domain to
            integer coordinates of dimension number of features
        :param data: NxD dimensional matrix (D=self.dimension)
        :return: Nxd dimensional integer matrix (d=self.numberOfFeatures)
        """
        data = data.view(np.float)
        encoding = []
        start = 0
        for featureIndex, label in enumerate(self.labels):
            d = self.dimensionOfFeature(label, featureIndex)
            encoding.append(self.toIndicesFeature(label,
                                                  data[:, start:start+d],
                                                  featureIndex))
            start += d
        return np.array(encoding).T

    def fromIndices(self, data):
        """
            converts given integer coordinates to points in the domain
        :param data: Nxd dimensional integer matrix (d=self.numberOfFeatures)
        :return: NxD dimensional matrix (D=self.dimension)
        """
        decoding = []
        for featureIndex, label in enumerate(self.labels):
            decoding.append(self.fromIndicesFeature(label,
                                                  data[:, featureIndex],
                                                  featureIndex))
        return np.hstack(decoding)

    def compress(self, data):
        """
            removes replicated rows
            returns reduced data matrix and count vector
        """
        newTypeSize = data.dtype.itemsize * data.shape[1]
        newType = np.dtype((np.void, newTypeSize))

        uniqueLines, counts = np.unique(np.ascontiguousarray(data).view(newType), return_counts=True)
        samples = uniqueLines.view(data.dtype).reshape(-1, data.shape[1]).astype(float)
        return samples, counts

    def __iter__(self):
        for uniqueValues in self.uniqueValues:
            yield uniqueValues
