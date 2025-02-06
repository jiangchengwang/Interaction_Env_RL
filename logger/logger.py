import logging
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime

# 颜色代码（仅适用于终端）
class LogColors:
    RESET = "\033[0m"
    RED = "\033[31m"        # ERROR, CRITICAL
    YELLOW = "\033[33m"     # WARNING
    GREEN = "\033[32m"      # INFO
    BLUE = "\033[34m"       # DEBUG
    MAGENTA = "\033[35m"    # OTHER

# 应用级别日志名称
APP_LOGGER_NAME: str = 'Interaction_Env_RL'

class ColoredFormatter(logging.Formatter):
    """
    自定义日志格式化类，实现颜色化日志输出
    """
    COLORS = {
        'DEBUG': LogColors.BLUE,
        'INFO': LogColors.GREEN,
        'WARNING': LogColors.YELLOW,
        'ERROR': LogColors.RED,
        'CRITICAL': LogColors.RED
    }

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, LogColors.MAGENTA)
        log_message = super().format(record)
        return f"{log_color}{log_message}{LogColors.RESET}"  # 加颜色并重置


def setup_app_level_logger(logger_name: str = APP_LOGGER_NAME,
                           level: str = 'DEBUG',
                           use_stdout: bool = True,
                           file_name: str = "debug.log",
                           max_file_size: int = 5 * 1024 * 1024,  # 5MB
                           backup_count: int = 3) -> logging.Logger:
    """
    创建一个应用级别的 Logger，支持日志轮转和追加模式。

    Args:
        logger_name (str, optional): Logger 名称，默认为 APP_LOGGER_NAME。
        level (str, optional): 日志级别，默认为 'DEBUG'。
        use_stdout (bool, optional): 是否输出到终端，默认为 True。
        file_name (str, optional): 日志文件名，默认为 "debug.log"。
        max_file_size (int, optional): 单个日志文件的最大大小（单位：字节），默认为 5MB。
        backup_count (int, optional): 备份的日志文件数量，默认为 3。

    Returns:
        logging.Logger: 配置好的 logger 实例
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # 日志格式（包含时间）
    log_format = "[%(asctime)s] [%(levelname)-8s] %(filename)s:%(funcName)s [Line %(lineno)d] - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(log_format, datefmt=date_format)

    # 轮转文件日志处理器（追加模式）
    file_handler = RotatingFileHandler(
        file_name, mode='a', maxBytes=max_file_size, backupCount=backup_count, encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 终端日志处理器（带颜色）
    if use_stdout:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(ColoredFormatter(log_format, datefmt=date_format))
        logger.addHandler(stdout_handler)

    return logger


def get_logger(module_name: str) -> logging.Logger:
    """
    获取模块的子 logger

    Args:
        module_name (str): 模块名

    Returns:
        logging.Logger: 配置好的 logger 实例
    """
    return logging.getLogger(APP_LOGGER_NAME).getChild(module_name)