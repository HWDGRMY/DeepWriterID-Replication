import os
import sys
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.training.trainer import train

if __name__ == '__main__':
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'configs', 'default.yaml'
    )
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    train(config)