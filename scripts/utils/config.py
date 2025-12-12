import os
from pathlib import Path
import yaml
from dotenv import load_dotenv
from datetime import datetime


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__


def _to_dotdict(x):
    return DotDict({k: _to_dotdict(v) if isinstance(v, dict) else v for k, v in x.items()})


load_dotenv()


PROJECT_ID = os.getenv("PROJECT_ID", "p00")
PROJECT_NAME = os.getenv("PROJECT_NAME", "test_project")

FILE_PATH_INPUT = os.getenv("FILE_PATH_INPUT", "./data")
FILE_PATH_OUTPUT = os.getenv("FILE_PATH_OUTPUT", "./results")

PARAMS_ID = os.getenv("PARAMS_ID", "000")
PARAMS_NAME = os.getenv("PARAMS_NAME", "default")


yml_candidates = list(Path("params").glob(f"{PARAMS_ID}_*.yml"))
if not yml_candidates:
    raise FileNotFoundError(f"No params YAML found for ID {PARAMS_ID} in ./params")
with open(yml_candidates[0], "r", encoding="utf-8") as f:
    yml_cfg = _to_dotdict(yaml.safe_load(f) or {})

# Generating output directory
try:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")  # avoid ':' for Windows
    OUTPUT_DIR = os.path.join(FILE_PATH_OUTPUT, f"output_{PARAMS_ID}_{stamp}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
except Exception as e:
    print(f"Failed to create output directory: {e}")


cfg = DotDict(
    {
        "PROJECT_ID": PROJECT_ID,
        "PROJECT_NAME": PROJECT_NAME,

        "PARAMS_ID": PARAMS_ID,
        "PARAMS_NAME": PARAMS_NAME,

        "FILE_PATH_INPUT": FILE_PATH_INPUT,
        "OUTPUT_DIR": OUTPUT_DIR,
        
        **yml_cfg,
    }
)