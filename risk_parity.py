import pandas as pd

def get_risk_parity_weights(predicted_vol):

    inverse_vol = 1.0 / predicted_vol
    weights = inverse_vol.div(inverse_vol.sum(axis=1), axis=0)

    return weights