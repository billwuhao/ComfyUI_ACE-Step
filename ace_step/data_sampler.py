import json
from pathlib import Path
import random
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT_DIR = f"{current_dir}/examples/"


class DataSampler:
    def __init__(self, root_dir=DEFAULT_ROOT_DIR):
        self.root_dir = root_dir
        self.input_params_files = list(Path(self.root_dir + "input_params").glob("*.json")) + \
                                  list(Path(self.root_dir + "zh_rap_lora").glob("*.json"))

    def load_json(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def sample(self):
        json_path = random.choice(self.input_params_files)
        json_data = self.load_json(json_path)
        return json_data
