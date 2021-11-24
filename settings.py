from enum import Enum

class PosFilter(str, Enum):
    NO = "no"
    POS = "pos"

class CqrSettings:
    verbose: bool = False


class HqeSettings(CqrSettings):
    """Settings for HQE with defaults tuned on CAsT"""

    M: int = 5  # number of aggregate historical queries
    eta: float = 10.0  # QPP threshold for first stage retrieval
    R_topic: float = 4.5  # topic keyword threshold
    R_sub: float = 3.5  # subtopic keyword threshold
    filter: PosFilter = PosFilter.POS  # 'no' or 'pos'


class NtrSettings(CqrSettings):
    """Settings for T5 model for NTR"""

    model_name: str = "castorini/t5-base-canard"
    max_length: int = 64
    num_beams: int = 10
    early_stopping: bool = True
    N: int = 2
    
class NtrBartSettings(CqrSettings):
    """Settings for BART model for NTR"""

    model_path: str = "../input/bartmodels/models/bart-summarizer/checkpoint-6500/"
    max_length: int = 64
    num_beams: int = 5
    early_stopping: bool = True
    N: int = 2

class PipelineSettings:
    """Settings for the pipeline"""

    top_k: int = 10
    early_fusion: bool = True