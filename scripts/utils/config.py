import os
from pathlib import Path
import yaml
from dotenv import load_dotenv


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def _to_dotdict(x):
    return DotDict({k: _to_dotdict(v) if isinstance(v, dict) else v for k, v in x.items()})


load_dotenv()

FILE_PATH_INPUT = os.getenv("FILE_PATH_INPUT", "./data")
FILE_PATH_OUTPUT = os.getenv("FILE_PATH_OUTPUT", "./results")
PARAMS_ID = os.getenv("PARAMS_ID", "000")

yml_candidates = list(Path("params").glob(f"{PARAMS_ID}_*.yml"))
if not yml_candidates:
    raise FileNotFoundError(f"No params YAML found for ID {PARAMS_ID} in ./params")
with open(yml_candidates[0], "r", encoding="utf-8") as f:
    yml_cfg = _to_dotdict(yaml.safe_load(f) or {})

cfg = DotDict(
    {
        "FILE_PATH_INPUT": FILE_PATH_INPUT,
        "FILE_PATH_OUTPUT": FILE_PATH_OUTPUT,
        "PARAMS_ID": PARAMS_ID,
        **yml_cfg,
    }
)

# import os
# import yaml
# from pathlib import Path
# from dotenv import load_dotenv

# env = load_dotenv()
# def get_bool_env(var_name, default=False):
#     return os.getenv(var_name, str(default)).lower() in ("1", "true", "yes", "on")

# FILE_PATH_INPUT = os.getenv("FILE_PATH_INPUT")
# # FILE_NAME_INPUT = os.getenv("FILE_NAME_INPUT")
# FILE_PATH_OUTPUT = os.getenv("FILE_PATH_OUTPUT")
# # FILE_NAME_OUTPUT = os.getenv("FILE_NAME_OUTPUT")
# # FILE_NAME_BATCH_INFO = os.getenv("FILE_NAME_BATCH_INFO")

# PARAMS_ID = int(os.getenv("PARAMS_ID", "000"))

# # SAMPLE_PREFIX = os.getenv("SAMPLE_PREFIX", "BIO")
# # FEATURE_ID_COLNAME = os.getenv("FEATURE_ID_COLNAME", "row ID")

# # IS_COL = os.getenv("IS_COL", "11974")
# # DEFAULT_ZERO_THRESHOLD = float(os.getenv("DEFAULT_ZERO_THRESHOLD", 0.2))
# # SAVE_INTERMEDIATE = get_bool_env("SAVE_INTERMEDIATE", True)

# # LOGGING_ENABLED = get_bool_env("LOGGING_ENABLED", False)
# # LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")

# # get vars from yaml file "../params/{PARAMS_ID}_*.yml"

# def load_yaml_params(yml_path: str | Path) -> dict:
#     """
#     Load experiment parameters from a YAML file.

#     Args:
#         yml_path (str | Path): path to YAML configuration file.

#     Returns:
#         dict: parsed configuration dictionary with all sections.
#     """
#     yaml.load(open(yml_path, "r"), Loader=yaml.FullLoader)