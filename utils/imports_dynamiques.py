import importlib
import sys
from pathlib import Path

# === paths ===
def _find_project_root() -> str:
    """Remonte l'arborescence depuis le CWD jusqu'à trouver pyproject.toml."""
    for parent in [Path.cwd()] + list(Path.cwd().parents):
        if (parent / "pyproject.toml").exists():
            return str(parent)
    return str(Path.cwd())

dir_level = _find_project_root()
if dir_level not in sys.path:
    sys.path.insert(0, dir_level)

# === packages ===
import utils
import metrics
import models
import networks
import pipelines

# === utils ===
import utils.utils
import utils.data_simulate
import utils.data_preprocessing
import utils.plots_timeseries
import utils.plots_training

from utils.utils import *
from utils.data_simulate import *
from utils.data_preprocessing import *
from utils.plots_timeseries import *
from utils.plots_training import *

# === metrics ===
import metrics.MLE_params
import metrics.global_stats
import metrics.discriminative_score
import metrics.predictive_score

from metrics.MLE_params import *
from metrics.global_stats import *
from metrics.discriminative_score import *
from metrics.predictive_score import *

# === models ===
import models.schedules
import models.loss
import models.sampler
import models.trainer

from models.schedules import *
from models.loss import *
from models.sampler import *
from models.trainer import *

# === networks ===
import networks.attention
import networks.conv
import networks.norm
import networks.residual
import networks.time_embedding
import networks.transformers
import networks.unets

from networks.attention import *
from networks.conv import *
from networks.norm import *
from networks.residual import *
from networks.time_embedding import *
from networks.transformers import *
from networks.unets import *

# === pipelines ===
import pipelines.utils_pipelines
import pipelines.dicts
import pipelines.pipeline_data
import pipelines.pipeline_schedule
import pipelines.pipeline_training
import pipelines.pipeline_sampling
import pipelines.pipeline_metrics
import pipelines.pipeline

from pipelines.utils_pipelines import *
from pipelines.dicts import *
from pipelines.pipeline_data import *
from pipelines.pipeline_schedule import *
from pipelines.pipeline_training import *
from pipelines.pipeline_sampling import *
from pipelines.pipeline_metrics import *
from pipelines.pipeline import *

# === reloads ===
for m in [
    utils.utils,
    utils.data_simulate,
    utils.data_preprocessing,
    utils.plots_timeseries,
    utils.plots_training,
    metrics.MLE_params,
    metrics.global_stats,
    metrics.discriminative_score,
    metrics.predictive_score,
    models.schedules,
    models.loss,
    models.sampler,
    models.trainer,
    networks.attention,
    networks.conv,
    networks.norm,
    networks.residual,
    networks.time_embedding,
    networks.transformers,
    networks.unets,
    pipelines.utils_pipelines,
    pipelines.dicts,
    pipelines.pipeline_data,
    pipelines.pipeline_schedule,
    pipelines.pipeline_training,
    pipelines.pipeline_sampling,
    pipelines.pipeline_metrics,
    pipelines.pipeline,
]:
    importlib.reload(m)

# === re-export after reload ===
from utils.utils import *
from utils.data_simulate import *
from utils.data_preprocessing import *
from utils.plots_timeseries import *
from utils.plots_training import *

from metrics.MLE_params import *
from metrics.global_stats import *
from metrics.discriminative_score import *
from metrics.predictive_score import *

from models.schedules import *
from models.loss import *
from models.sampler import *
from models.trainer import *

from networks.attention import *
from networks.conv import *
from networks.norm import *
from networks.residual import *
from networks.time_embedding import *
from networks.transformers import *
from networks.unets import *

from pipelines.utils_pipelines import *
from pipelines.dicts import *
from pipelines.pipeline_data import *
from pipelines.pipeline_schedule import *
from pipelines.pipeline_training import *
from pipelines.pipeline_sampling import *
from pipelines.pipeline_metrics import *
from pipelines.pipeline import *

__all__ = [name for name in globals() if not name.startswith("_")]