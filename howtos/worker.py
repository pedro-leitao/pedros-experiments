
import os
import json

import numpy as np
import tensorflow as tf
from tensorflow.config import set_visible_devices
import model_setup
import env_setup

def worker_setup():
    # Setup for distributed training
    env_setup.set_env()
    # Hide GPU from visible devices
    set_visible_devices([], 'GPU')

def run_worker():
    # Set up a MultiWorkerMirroredStrategy
    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    # Assume global batch size and calculate per-replica batch size
    GLOBAL_BATCH_SIZE = 50
    # Build dataset with the specified global batch size
    dataset = model_setup.build_dataset(strategy, GLOBAL_BATCH_SIZE)
    # Calculate steps per epoch based on your actual dataset size
    # Here, 1000 is the number of samples in the dataset
    STEPS_PER_EPOCH = 1000 // GLOBAL_BATCH_SIZE

    with strategy.scope():
        # Model building/compiling need to be within `strategy.scope()`.
        multi_worker_model = model_setup.build_model()

    multi_worker_model.fit(dataset, epochs=10, steps_per_epoch=STEPS_PER_EPOCH)
    print("Exiting worker...")

    return multi_worker_model, dataset
