import numpy as np

def calculate_mape(y_true, y_pred):
    """
    Returns the Mean Absolute Percentage Error.
    Use this to quantify how 'wrong' the model is for the presentation.
    """
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100