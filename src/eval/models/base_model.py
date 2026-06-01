import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)
from typing import Dict, Any

import numpy as np
import av

import pytorch_lightning as pl

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`list[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

class BaseModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super(BaseModel, self).__init__()
        self.params = kwargs
        self.save_hyperparameters()

    def create_prompt(self, conversation: Dict[str, Any]):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def post_process_response(self, response: str) -> str:
        """
        Post-process the response from the model.
        This can include cleaning up the response, formatting, etc.
        """
        return response.strip()