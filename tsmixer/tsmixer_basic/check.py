import tensorflow as tf
import pandas as pd
# Load the saved model
model = tf.keras.models.load_model(r'C:\Users\vishn\OneDrive\Documents\tsmixer\tsmixer\tsmixer_basic\model.h5')

# Make predictions
input_data = pd.read_csv(r'C:\Users\vishn\OneDrive\Documents\tsmixer\sample_normalized.csv')
predictions = model.predict(input_data)
print(predictions)

