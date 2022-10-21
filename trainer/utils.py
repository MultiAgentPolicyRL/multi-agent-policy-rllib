import argparse
import json
import logging
import os
from datetime import datetime

import yaml


def get_basic_logger(logger_name="default"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

logger = get_basic_logger("UTILS")

def process_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=str, help="Path to the directory for this run.", default="phase1",)
    parser.add_argument("--pw", type=str, help="Redis password, used only when clustering", default="password",)
    parser.add_argument("--ip_address", type=str, help="Ray ip:port, used only when clustering", default="",)
    parser.add_argument("--cluster", type=bool, help="If experiment is running on a cluster set to `True`, otherwise don't use", default=False, )
    args = parser.parse_args()
    
    config_path = os.path.join(args.run_dir, "config.yaml")

    if not os.path.isdir(args.run_dir):
        logger.error(f"'{args.run_dir}' is not a directory.")
    if not os.path.isfile(config_path):
        logger.error(f"'{config_path}' does not exist.")

    # with open(config_path, "r") as f:
    #     configuration = yaml.safe_load(f)
    #     logger.info(f"Loaded '{config_path}'.")

    return args.run_dir, None, args.pw, f"{args.ip_address}", args.cluster

def set_up_dirs(output_folder:str=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Outputs")):
    ### Set up directories for logging and saving. Restore if not completed
    date_directory = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dirname = os.path.join(output_folder, date_directory)

    ### If the output directory does not exist, create it and return the path and False since first run
    if not os.path.isdir(output_folder):
        os.makedirs(dirname)
        logger.info("Created output directory: '{}'".format(dirname))
        return dirname, False
    
    ### If the output directory exists, check if the last data directory is completed
    date_directories = list(os.listdir(output_folder))
    if '.DS_Store' in date_directories:
        date_directories.remove('.DS_Store')
    date_directories.reverse()

    for dir in date_directories:
        agent_tmp_file = os.path.join(output_folder, dir, "Agents.json")
        if os.path.isfile(agent_tmp_file):
            with open(agent_tmp_file, 'r') as f:
                data = {"completed":True}#data = json.load(f)
            if not data['completed']:
                logger.info("Found incomplete run. Restoring from: '{}'".format(dir))
                return os.path.join(output_folder, dir), True
        
    ### If the last data directory is completed, create a new one
    os.makedirs(dirname)
    logger.info("No incomplete run found. Created output directory: '{}'".format(dirname))
    return dirname, False

def save_data(file:str, data:dict):
    ### Save data to file
    with open(file, 'w') as f:
        json.dump(data, f, indent=4)

    logger.info("Saved data to: '{}'".format(file))