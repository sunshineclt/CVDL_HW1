import keras
from keras.callbacks import ModelCheckpoint

from DataGenerator import DirectoryIterator
from config import *
from im_utils import func_batch_handle_with_multi_process, recycle_pool
from tensorboard import StepTensorBoard
from utils import context_creator, parse_weight_file, get_best_weights


class BaseClassifier(object):
    def __init__(self, name, image_size, learning_rate=1e-3, batch_size=BATCH_SIZE):
        self.name = name
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weights = None
        self.optimizer = None

        self.context = context_creator(self.name)
        self.path_summary = self.context["summary"]
        self.path_weights = self.context["weights"]

        self.model = self.build_model()
        self._compiled = False

    def build_model(self):
        model = self.create_model()
        self.weights = get_best_weights(os.path.dirname(self.path_weights))
        if self.weights:
            model.load_weights(self.weights)
            print("Load %s successfully. " % self.weights)
        else:
            print("Model params not found. Thus start training from zero. ")
        return model

    def create_model(self):
        raise NotImplementedError("Don't create model for BaseClassifier")

    def compile_model(self, force_compile=False):
        if not self._compiled or force_compile:
            if not self.optimizer:
                self.optimizer = keras.optimizers.Adam(self.learning_rate)
            self.model.compile(loss="categorical_crossentropy", optimizer=self.optimizer, metrics=["accuracy"])
            self._compiled = True

    def data_generator(self, path, train=True, random_prob=0.5, **kwargs):
        return DirectoryIterator(path, None, target_size=(self.image_size, self.image_size), batch_size=self.batch_size,
                                 class_mode="categorical", batch_handler=lambda x: func_batch_handle_with_multi_process(x, train, random_prob),
                                 **kwargs)

    def train(self, **kwargs):
        # calculate files number
        steps_train = NUMBER_OF_TRAIN_IMAGE // self.batch_size
        print('Steps number is %d every epoch.' % steps_train)
        steps_val = NUMBER_OF_VAL_IMAGE // self.batch_size

        # build data generator
        train_generator = self.data_generator(PATH_TRAIN_IMAGE)
        val_generator = self.data_generator(PATH_VAL_IMAGE, train=False)

        # compile model if not
        self.compile_model()

        # start training
        if not os.path.exists(os.path.dirname(self.path_weights)):
            os.makedirs(os.path.dirname(self.path_weights))
        weights_info = parse_weight_file(self.weights) if self.weights else None
        init_epoch = weights_info[0] if weights_info else 0
        print('Start training from %d epoch.' % init_epoch)
        init_step = init_epoch * steps_train
        try:
            self.model.fit_generator(
                train_generator,
                steps_per_epoch=steps_train,
                callbacks=[
                    ModelCheckpoint(self.path_weights, verbose=1),
                    StepTensorBoard(self.path_summary, init_steps=init_step, write_freq_step=200),
                ],
                initial_epoch=init_epoch,
                epochs=EPOCH,
                validation_data=val_generator,
                validation_steps=steps_val,
                verbose=1,
                **kwargs
            )
        except KeyboardInterrupt:
            print('\nStop by keyboardInterrupt, try saving weights.')
            # model.save_weights(PATH_WEIGHTS)
            # print('Save weights successfully.')
        finally:
            recycle_pool()
