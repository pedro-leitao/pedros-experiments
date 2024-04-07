
import tensorflow as tf
from model_setup import predict, plot_dataset_and_predictions

saved_model = tf.keras.models.load_model("trained_model.keras")
saved_dataset = tf.data.Dataset.load("dataset.keras", element_spec=None)
pred_x, pred_y = predict(saved_model, 100)
plot_dataset_and_predictions(saved_dataset, pred_x, pred_y)
