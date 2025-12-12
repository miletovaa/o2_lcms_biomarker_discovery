import os
import logging

from utils.config import cfg

import s01prep
import s02rforest


# Step 0.1: Loading mode and environment variables from .env file
OUTPUT_DIR = cfg.get("OUTPUT_DIR", "./results")

# Step 0.2: Setting up logging
# def set_logging(name: str):
#     if cfg.logging.enabled:
#         logging_path = os.path.join(OUTPUT_DIR, 'logs')
#         os.makedirs(logging_path, exist_ok=True)
#         log_handlers = [logging.StreamHandler()]
#         log_file_path = os.path.join(logging_path, f'{name}.log')
#         log_handlers.append(logging.FileHandler(log_file_path))
#         logging.basicConfig(
#             level=logging.INFO, 
#             handlers=log_handlers,
#             format='%(asctime)s - %(levelname)s - %(message)s',
#             datefmt='%d-%m-%Y %H:%M:%S'
#         )
#         logging.info(f"Logging is set up successfully. Starting data preparation.")


# Step 1.1: Data Preparation (RP)
# set_logging("s01prep_rp")
output_rp = s01prep.main("rp", cfg)


# Step 1.2: Data Preparation (ANP)
# set_logging("s01prep_anp")
output_anp = s01prep.main("anp", cfg)


# # Step 2: Random Forest Modeling
# set_logging("s02rforest")
s02rforest.main(cfg, output_rp=output_rp, output_anp=output_anp)