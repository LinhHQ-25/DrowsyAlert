"""Experimental binocular CNN decision; EAR remains an observed feature."""


def cnn_eye_decision(probabilities, smooth_probability, was_closed):
    """Return (closed, confident, rule). Thresholds require webcam validation."""
    if len(probabilities) != 2:
        return False, False, "cnn_missing_eye"
    if min(probabilities) >= 0.60 and smooth_probability >= 0.72:
        return True, True, "cnn_closed"
    if was_closed and min(probabilities) >= 0.45 and smooth_probability >= 0.55:
        return True, True, "cnn_hold_closed"
    if max(probabilities) <= 0.50 and smooth_probability <= 0.35:
        return False, True, "cnn_open"
    return False, False, "cnn_uncertain"
