# coding=utf-8
import os
import csv

import config
from BaseClassifier import BaseClassifier
from ClassifierInceptionResnetV2 import ClassifierInceptionResnetV2
from im_utils import *


# predictions = model.predict(inputs)

def parse_prediction(files, predictions, top=3, return_with_prob=False):
    result = np.argsort(predictions)
    result = result[:, -top:][:, ::-1]
    assert len(files) == len(result)
    if return_with_prob:
        return [[(j, predictions[i][j]) for j in r] for i, r in enumerate(result)]
    else:
        return list(map(lambda x: x.tolist(), result))


class Predictor:
    def __init__(self, func_predict, target_size, mode=None, batch_handler=None):
        self.func_predict = func_predict
        self.target_size = target_size
        self.mode = mode
        self.batch_handler = batch_handler

    def __call__(self, files, top=3, return_with_prob=False, **kwargs):
        if isinstance(files, str):
            files = [files]
        predictions = self.perform_predict(files, **kwargs)
        return parse_prediction(files, predictions, top, return_with_prob)

    def perform_predict(self, files, **kwargs):
        inputs, patch = im2array(files, self.target_size, self.mode)
        assert patch * len(files) == len(inputs)
        if self.batch_handler:
            inputs = self.batch_handler(inputs)
        predictions = self.func_predict(inputs, **kwargs)
        if patch != 1:
            predictions = np.array([np.mean(predictions[i: i + patch], axis=0) for i in range(0, len(inputs), patch)])
        return predictions


class KerasPredictor(Predictor):
    def __init__(self, classifier, mode=None, batch_handler=None):
        assert isinstance(classifier, BaseClassifier), \
            'The classifier is not a instance of %s' % (type(BaseClassifier))
        self.model = classifier.model
        self.weights = classifier.weights
        self.name = classifier.name
        # set default batch_handler if not exists
        if not batch_handler:
            batch_handler = lambda x: func_batch_handle_with_multi_process(x, False)
        h, w = self.model.input_shape[1:3]
        assert h == w, 'Width is not equal with height.'
        Predictor.__init__(self, self.model.predict, w, mode, batch_handler)


if __name__ == '__main__':
    predictor = KerasPredictor(ClassifierInceptionResnetV2("Inception_Resnet_V2"), 'val')
    transform_array = [0]
    for i in range(1, 8):
        transform_array.append(i)
        for j in range(0, 10):
            transform_array.append(int(str(i)+str(j)))
    transform_array.extend([8, 9])

    with open(os.path.join(config.PATH_BASE, "test.info"), "r") as rf, open(os.path.join(config.PATH_BASE, "submit.info"), "w") as wf:
        reader = csv.DictReader(rf, delimiter=" ", fieldnames=["file"])
        writer = csv.DictWriter(wf, delimiter=" ", fieldnames=["file", "1_predict", "2_predict", "3_predict"])
        for row in reader:
            path = os.path.join(config.PATH_BASE, row["file"])
            prediction = predictor(path, return_with_prob=True)

            print(row["file"].split("/")[1], prediction)
            writer.writerow({"file": row["file"],
                             "1_predict": transform_array[prediction[0][0][0]],
                             "2_predict": transform_array[prediction[0][1][0]],
                             "3_predict": transform_array[prediction[0][2][0]]})
    # path = os.path.join(config.PATH_VAL_IMAGE, "12/004201.jpg")
    # prediction = predictor(path, return_with_prob=True)
    # print(prediction)
