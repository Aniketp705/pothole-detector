import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def benchmark():
    print("Loading model...")
    try:
        model = load_model('pothole_detector_final.h5')
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Generate dummy input (1 image, 224x224, 3 channels)
    dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)

    print("Warming up...")
    # Warmup runs to initialize graph
    for _ in range(5):
        model.predict(dummy_input, verbose=0)

    print("Running benchmark (50 iterations)...")
    latencies = []
    for _ in range(50):
        start_time = time.time()
        model.predict(dummy_input, verbose=0)
        end_time = time.time()
        latencies.append((end_time - start_time) * 1000) # Convert to ms

    avg_latency = sum(latencies) / len(latencies)
    print(f"\nResults:")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Min Latency: {min(latencies):.2f} ms")
    print(f"Max Latency: {max(latencies):.2f} ms")

if __name__ == "__main__":
    benchmark()
