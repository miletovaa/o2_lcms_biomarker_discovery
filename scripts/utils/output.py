from typing import Literal
import os
import pandas as pd
from datetime import datetime


OutputType = Literal[
    "raw",
    "prep",
    "norm",
    "stats",
    "ml",
    "viz",
    "fig",
    "output",
    "report",
]

def create_output_file_name(step: str, type: OutputType, description: str, ext: str, cfg, output_dir: str) -> str:
    date = pd.Timestamp.now().strftime("%Y%m%d")
    return os.path.join(output_dir, f"{cfg.get("PROJECT_ID")}_{cfg.get("PROJECT_NAME")}_{cfg.get("PARAMS_ID")}_{cfg.get("PARAMS_NAME")}_{step}_{type}_{description}_{date}.{ext}")


def save_figure(fig, filename, show=False):
    fig.write_image(filename)
    fig.write_html(filename.replace(".png", ".html"))
    print(f"Figure saved to {filename}")
    if show is True:
        fig.show()