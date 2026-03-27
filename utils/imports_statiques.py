# utils/imports_statiques.py

# === Bibliothèque standard ===
import os
import sys
import time
import math
import json
import random
import hashlib
import itertools
import copy
from abc import ABC, abstractmethod
from collections import namedtuple
from functools import partial
from typing import List, Optional, Tuple

# === Dépendances tierces ===
import numpy as np
import pandas as pd
pd.set_option("display.max_colwidth", None)
pd.set_option("display.width", None)
pd.set_option("display.max_columns", None)

# === PyTorch ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, TensorDataset
import torchaudio.transforms as T

# torchvision / diffusers (optionnels selon usage)
from torchvision import transforms as tv_transforms
from torchvision.datasets import MNIST
from diffusers import UNet1DModel

# === Outils tensor / manipulation ===
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

# === SciPy / stats ===
from scipy import integrate
from scipy.optimize import minimize
from scipy.special import logit
from scipy.stats import ncx2, gaussian_kde, wasserstein_distance

# === Progression / affichage ===
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from tqdm.auto import tqdm, trange

# === Plotly ===
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import HTML, display
