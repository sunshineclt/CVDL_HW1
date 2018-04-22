from BaseClassifier import BaseClassifier
from config import *
from keras.applications import InceptionResNetV2
from keras.layers import Dense
from keras.engine import Model


class ClassifierInceptionResnetV2(BaseClassifier):
    def __init__(self, name="inception_resnet_v2", learning_rate=1e-3, batch_size=BATCH_SIZE):
        BaseClassifier.__init__(self, name, IM_SIZE_299, learning_rate, batch_size)

    def create_model(self):
        weights = "imagenet" if self.context['load_imagenet_weights'] else None
        model_inception_resnet_v2 = InceptionResNetV2(include_top=False, weights=weights,
                                                      input_shape=(self.image_size, self.image_size, 3), pooling='avg')
        for layer in model_inception_resnet_v2.layers[:-50]:
            layer.trainable = False
        now = model_inception_resnet_v2.output
        now = Dense(CLASS_NUMBER, activation="softmax")(now)
        model = Model(input=model_inception_resnet_v2.input, outputs=now)
        return model

    def data_generator(self, path, train=True, random_prob=1, **kwargs):
        return BaseClassifier.data_generator(self, path, train, random_prob, **kwargs)


if __name__ == "__main__":
    classifier = ClassifierInceptionResnetV2("Inception_Resnet_V2", learning_rate=1e-3)
    classifier.train()
