"""Constants for the model.

These values can be tweaked to affect model training performance.
"""

import os

LEARNING_RATE = 1e-3

# Notice that the model is fit using a larger than default batch size of 2048,
# this is important to ensure that each batch has a decent chance of
# containing a few positive samples.
TRAIN_BATCH_SIZE = 2048
EVAL_BATCH_SIZE = 2048

# Nr of epochs to run
NUM_EPOCHS = 2

INITIAL_BIAS = None

INITIAL_WEIGHTS = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                               '..',  # models dir
                                               '..',  # src dir
                                               '..',  # root dir
                                               'config/initial_weights/initial_weights'))  # noqa: E501


# CLASS_WEIGHT = None
# NOTE: Uncoment below to use class weights.
# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
# weight_for_0 = (1 / neg) * (total / 2.0)
# weight_for_1 = (1 / pos) * (total / 2.0)
# class_weight = {0: weight_for_0, 1: weight_for_1}

CLASS_WEIGHT = {0: 0.5071813487858222, 1: 35.31240188383046}
