"""日志工具"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_str: Optional[str] = None,
    ddp_aware: bool = True
) -> logging.Logger:
    """设置日志器，支持DDP
    
    Args:
        name: logger名称
        log_file: 日志文件路径，None表示不保存到文件
        level: 日志级别
        format_str: 日志格式字符串
        ddp_aware: 是否支持DDP，仅在rank 0输出到控制台
        
    Returns:
        logger: 配置好的logger
        
    Examples:
        >>> logger = setup_logger('GeoMetricLab', ddp_aware=True)
        >>> logger.info('Training started')
    """
    if format_str is None:
        format_str = "[%(levelname)s] %(name)s: %(message)s"
    
    formatter = logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S")
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # 清除已有的handlers
    logger.propagate = False  # 防止重复输出
    
    # 检查是否在DDP环境中
    rank = 0
    if ddp_aware:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
        except ImportError:
            pass
    
    # 控制台handler（仅在rank 0或非DDP模式下）
    if rank == 0 or not ddp_aware:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件handler（所有rank都写入，但使用不同的文件名）
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # DDP模式下，每个rank写入不同的文件
        if ddp_aware and rank > 0:
            log_file = log_file.parent / f"{log_file.stem}_rank{rank}{log_file.suffix}"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_rank_zero_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """获取rank 0专用的logger（简化版）
    
    Args:
        name: logger名称
        log_file: 日志文件路径
        level: 日志级别
        
    Returns:
        logger: 配置好的logger，仅在rank 0输出
    """
    return setup_logger(name, log_file, level, ddp_aware=True)


def print_rank_0(message: str):
    """仅在Rank 0打印信息"""
    import torch.distributed as dist
    import os
    
    # Check for DDP environment variables or initialized group
    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    elif "RANK" in os.environ:
        rank = int(os.environ["RANK"])
    elif "LOCAL_RANK" in os.environ:
        rank = int(os.environ["LOCAL_RANK"])

    if rank == 0:
        print(message)
