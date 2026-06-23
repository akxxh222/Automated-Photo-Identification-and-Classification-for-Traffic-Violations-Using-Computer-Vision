from .preprocessor import FramePreprocessor
_preprocessor = FramePreprocessor()

def preprocess_frame(frame):
    proc, _ = _preprocessor.process_frame(frame, 0)
    return proc
