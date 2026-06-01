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


question_type = "temporal_loc"
question_templates = dict()
question_templates["1part_order"] = """Which of the highlighted parts are connected (brought in direct physical contact) to {query_part} next starting from the assembly state shown in the image?"""
question_templates["1part_order_last"] = """Which highlighted part is the last to connect (brought in direct physical contact) to {query_part} in the video?"""
question_templates["1edge_order"] = """Which highlighted pair of parts is connected first in the video?"""
question_templates["1edge_order_last"] = """Which highlighted pair of parts is connected last in the video?"""
question_templates["latest_change"] = """Which labeled part was connected last to reach the current assembly state shown in the image?"""
question_templates["next_change"] = """Which labeled part will be connected next to the current assembly state shown in the image?"""

template_option_type = dict()
template_option_type["1part_order"] = "part"
template_option_type["1part_order_last"] = "part"
template_option_type["1edge_order"] = "edge"
template_option_type["1edge_order_last"] = "edge"
template_option_type["latest_change"] = "part"
template_option_type["next_change"] = "part"
