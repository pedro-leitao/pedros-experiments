
import tensorflow as tf
from worker import worker_setup, run_worker

worker_setup()
trained_model, dataset = run_worker()
print("Worker finished...")
trained_model.save("trained_model.keras")
tf.data.Dataset.save(dataset, "dataset.keras")
