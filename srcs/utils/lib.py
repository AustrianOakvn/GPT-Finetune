import os
import time
import signal
import datetime
from abc import abstractmethod, ABCMeta
from dataclasses import dataclass
import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf

import pandas as pd
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed(seed_val)