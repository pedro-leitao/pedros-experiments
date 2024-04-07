
import os
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.data import Dataset

def build_dataset(strategy, global_batch_size):
  # Generate a dataset
  np.random.seed(42)  # For reproducibility
  x = np.linspace(-1, 1, 1000)  # 1000 linearly spaced numbers
  y = 2 * x + 1 + np.random.normal(scale=0.1, size=x.shape)  # Linear equation with noise

  # Create a TensorFlow dataset
  dataset = Dataset.from_tensor_slices((x[:, None], y))
  # Shuffle, batch, and repeat
  dataset = dataset.shuffle(buffer_size=1000).batch(global_batch_size).repeat()
    
  return dataset

def plot_dataset_and_predictions(dataset, pred_x, pred_y):
  import matplotlib.pyplot as plt
  for x, y in dataset.take(1):
    plt.scatter(x, y, label='Data')
  plt.scatter(pred_x, pred_y, label='Predictions')
  plt.legend()
  plt.show()

def build_model():
  input_layer = Input(shape=(1,))
  output_layer = Dense(units=1)(input_layer)
  model = Model(inputs=input_layer, outputs=output_layer)

  model.compile(optimizer=Adam(learning_rate=0.1),
                loss='mean_squared_error',
                metrics=['mae'])
  return model

# Predict n random values from -1 to 1 using the trained model and return the predictions
def predict(model, n):
  x = np.random.uniform(-1, 1, n)
  y = model.predict(x)
  return x, y
