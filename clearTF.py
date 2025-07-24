import gc
import tensorflow.keras.backend as K

print("🧹 Clearing TensorFlow memory...")
K.clear_session()
gc.collect()
print("✅ Memory cleared.")
