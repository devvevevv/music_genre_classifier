"""Backwards-compatible CLI helpers built on the reusable predictor."""

from predictor import predict


def analyze_audio_file(file_path, model_path=None):
    """Return the most likely genre for an audio file, or None on failure."""
    try:
        probabilities = predict(file_path, model_path=model_path)
    except Exception as e:
        print(f"Error analyzing audio file: {e}")
        return None

    return next(iter(probabilities), None)
