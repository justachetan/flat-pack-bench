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


question_type = "temporal_ord"
question_templates = dict()


question_templates["many_part_order"] = """Which of the following options lists the highlighted parts in the exact order that they become directly attached (physically in contact) to {query_part} in the video, from the first attachment event to the last?"""
# question_templates["edge_order"] = """Which of the following lists the pairs of parts in the exact sequence that they connect in the video, from the first connection event to the last?"""
question_templates["edge_order"] = """Each option below lists a sequence of part connections. Which of these options gets the temporal ordering of the connections correct?"""


template_option_type = dict()
template_option_type["many_part_order"] = "part_list"
template_option_type["edge_order"] = "edge_list"