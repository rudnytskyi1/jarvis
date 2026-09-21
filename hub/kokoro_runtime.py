"""Normalize ONNX inputs across the Kokoro v1 export formats."""
import numpy as np


class TypedSession:
    def __init__(self, session):
        self.session = session
        self._model_path = session._model_path

    def get_inputs(self):
        return self.session.get_inputs()

    def run(self, outputs, inputs):
        # kokoro-onnx 0.4.9 emits int32 speed for the input_ids export, although
        # the official v1 graph declares float. Honor the graph's actual types.
        types = {'tensor(float)': np.float32, 'tensor(int64)': np.int64, 'tensor(int32)': np.int32}
        values = {i.name: np.asarray(inputs[i.name], dtype=types[i.type]) for i in self.get_inputs()}
        return self.session.run(outputs, values)
