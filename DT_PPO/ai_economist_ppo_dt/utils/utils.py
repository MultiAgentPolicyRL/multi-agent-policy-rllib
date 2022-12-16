import os
import logging

from ai_economist import foundation
from ai_economist.foundation.base.base_env import BaseEnvironment

from ai_economist_ppo_dt.configs import env_config

class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold = "\x1b[1m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = '%(asctime)s | %(levelname)s | %(name)s --> %(message)s'
    debug_format = '%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d --> %(message)s'#'%(asctime)s | %(name)s | %(filename)s:%(lineno)d  | %(message)s'

    FORMATS = {
        logging.DEBUG: bold + debug_format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + debug_format + reset,
        logging.ERROR: red + debug_format + reset,
        logging.CRITICAL: bold_red + debug_format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%Y/%m/%d %H:%M:%S')
        return formatter.format(record)

def get_basic_logger(name, level=logging.DEBUG, log_path:str=None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = CustomFormatter()
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if log_path:
        par_dir = os.path.dirname(log_path)
        if not os.path.exists(par_dir):
            os.makedirs(par_dir)

        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s')
        fh = logging.FileHandler(log_path)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def create_environment(env_config: dict = env_config) -> BaseEnvironment:
    """
    Create an environment from a config dictionary.
    
    Parameters:
    ---
    env_config : dict
        config dictionary for the environment
    
    Returns:
    ---
    env : BaseEnvironment
        The environment
    """

    env:BaseEnvironment = \
        foundation.make_env_instance(
            **env_config['env_config_dict']
        )
    
    return env

def ms_to_time(ms):
    if isinstance(ms, float):
        ms = int(ms)

    milliseconds = str(ms)[-3:]

    while len(milliseconds) < 3:
        milliseconds = "0" + milliseconds

    ms = ms-int(milliseconds)
    seconds = int((ms/1000)%60)
    minutes = int((ms/(1000*60))%60)
    hours = int((ms/(1000*60*60))%24)

    if seconds < 10:
        seconds = f"0{int(seconds)}"
    if minutes < 10:
        minutes = f"0{int(minutes)}"
    if hours < 10 and hours > 0:
        hours = f"{int(hours)}"
    
    if int(hours) < 1:
        if int(minutes) < 1:
            if int(seconds) < 10:
                return f"{int(seconds)}.{milliseconds}"
            return f"{seconds}.{milliseconds}"
        elif int(minutes) < 10:
            return f"{int(minutes)}:{seconds}.{milliseconds}"
        return f"{minutes}:{seconds}.{milliseconds}"
    
    return f"{hours}:{minutes}:{seconds}.{milliseconds}"

time_logger = get_basic_logger("time_it")
def time_it(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        if int((end-start)*1000) > 0:
            time_logger.info(f"Time taken for {func.__name__} : {ms_to_time((end-start)*1000)}")
        # logger.debug(f"Time taken for {func.__name__} : {ms_to_time((end-start)*1000)}")
        return result
    return wrapper