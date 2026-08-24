# utils/metrics_utils.py

import time
import pandas as pd

# Measuring Inference Time
def measure_inference_time(prediction_function, image):
    start_time = time.perf_counter()
    result = prediction_function(image)

    end_time = time.perf_counter()
    inference_time = (end_time - start_time) * 1000

    return (result, round(inference_time, 2))


# Extracting Confidence
def get_confidence(result):
    if isinstance(result, dict):

        return result.get("confidence", 0)

    return 0


# Create Comparison Table
def create_comparison_table(model_results):
    return pd.DataFrame(model_results)


# Sorting By Confidence
def rank_models(model_results):
    ranked = sorted(model_results, key=lambda x: x["confidence"], reverse=True)

    return ranked


# Converting Confidence To Percentage
def confidence_percent(confidence):
    return round(confidence * 100, 2)


# Finding Best Model
def get_best_prediction(model_results):
    return max(model_results, key=lambda x: x["confidence"])