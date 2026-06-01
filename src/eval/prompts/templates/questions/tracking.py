from typing import Dict, List, Literal, Tuple, Any

import pyrootutils
root = pyrootutils.setup_root(
    search_from="./",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


import os
import sys
import os.path as osp
sys.path.append(osp.join(root, "src"))
import copy
import json
import math
import random
import string
import itertools

import numpy as np

from src.benchmark.generator.common import form_question_string, check_question_answer_string
from IKEAVideo.dataloader.assembly_video import decode_mask


question_type = "tracking"
question_templates = dict()
question_templates["track_multi"] = """Identify the correct set of matches from Image A to Image B."""
question_templates["track_single"] = """Identify which part in Image B matches {query_part} in Image A."""

template_option_type = dict()
template_option_type["track_multi"] = "edge_list"
template_option_type["track_single"] = "part"