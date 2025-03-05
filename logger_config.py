import os
import logging
import datetime
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_dir='logs', level=logging.INFO):
    """设置应用日志记录器"""
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件名，使用当前日期
    log_file = os.path.join(log_dir, f"{name}_{datetime.datetime.now().strftime('%Y%m%d')}.log")
    
    # 创建日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 创建文件处理器，最大10MB，保留5个备份
    file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setFormatter(formatter)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 获取或创建记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免处理器重复
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# 创建应用主日志记录器
app_logger = setup_logger('chat_app')
model_logger = setup_logger('model_operations')
api_logger = setup_logger('api_requests')
