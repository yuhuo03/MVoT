# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""V3D-CoT dataset for MVoT - Visual 3D Chain-of-Thought"""

import json
import os
import glob
from pathlib import Path

import datasets
from PIL import Image


_CITATION = """
@misc{v3d-cot,
    title={Visual 3D Chain-of-Thought: Extending MVoT to 3D Object Generation}, 
    author={Your Name},
    year={2024},
}
"""

_DESCRIPTION = """
Visual 3D Chain-of-Thought (V3D-CoT) dataset for 3D object assembly.
Each sample contains a goal instruction and progressive assembly steps with
interleaved verbal descriptions and visual states (rendered 2D images).
"""

_HOMEPAGE = ""

_LICENSE = ""

_DATA_DIR = r"sample_datasets"

_URLS = {
    "data_dir": _DATA_DIR,
}

# Instruction template for V3D-CoT
V3D_COT_INSTRUCTION = {
    "3d_assembly": "Goal: <GOAL_INSTRUCTION>\n\nResponse: <STEP_HISTORY>"
}


class V3DCotConfig(datasets.BuilderConfig):
    """BuilderConfig for V3D-CoT."""

    def __init__(self, tasks, modes, data_dir, **kwargs):
        """BuilderConfig for V3D-CoT.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(V3DCotConfig, self).__init__(**kwargs)
        self.tasks = tasks
        self.modes = modes
        self.data_dir = data_dir


class V3DCot(datasets.GeneratorBasedBuilder):
    """V3D-CoT dataset."""

    BUILDER_CONFIG_CLASS = V3DCotConfig
    BUILDER_CONFIGS = [
        V3DCotConfig(
            name="processed_v3d_cot",
            version=datasets.Version("0.0.0"),
            description=_DESCRIPTION,
            tasks=["3d_assembly"],
            modes=["single_step_visualization"],
            data_dir="sample_datasets"
        )
    ]

    DEFAULT_CONFIG_NAME = "processed_v3d_cot"

    def _info(self):
        features = datasets.Features(
            {
                'idx': datasets.Value('int32'),
                "input_text": datasets.Value("string"),
                "input_imgs": datasets.Sequence(datasets.Image()),
                "label_text": datasets.Value("string"),
                "label_imgs": datasets.Sequence(datasets.Image()),
                "label_img_paths": datasets.Sequence(datasets.Value("string")),
                "input_img_paths": datasets.Sequence(datasets.Value("string")),
                'task': datasets.Value('string'), 
                'train_task': datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        downloaded_files = _URLS

        tasks = self.config.tasks
        modes = self.config.modes
        data_dir = self.config.data_dir

        global _DATA_DIR_PREFIX
        _DATA_DIR_PREFIX = data_dir

        data_dirs = []
        for task in tasks:
            # Check if data_dir is the dataset folder itself (contains numbered subdirectories)
            if os.path.exists(data_dir) and os.path.isdir(data_dir):
                # Check if this directory contains numbered subdirectories (object folders)
                try:
                    subdirs = [d for d in os.listdir(data_dir) 
                              if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()]
                    if len(subdirs) > 0:
                        # This is the dataset folder itself
                        data_dirs.append(data_dir)
                    else:
                        # Try to find the dataset folder inside data_dir
                        potential_path = os.path.join(data_dir, downloaded_files['data_dir'])
                        if os.path.exists(potential_path):
                            data_dirs.append(potential_path)
                        else:
                            data_dirs.append(data_dir)
                except:
                    # If we can't list the directory, try the default path
                    potential_path = os.path.join(data_dir, downloaded_files['data_dir'])
                    if os.path.exists(potential_path):
                        data_dirs.append(potential_path)
                    else:
                        data_dirs.append(data_dir)
            else:
                # Default: append the dataset folder name
                data_dirs.append(os.path.join(data_dir, downloaded_files['data_dir']))
                  
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_dirs": data_dirs,
                    "split": "train",
                    "modes": modes
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split('dev'),
                gen_kwargs={
                    "data_dirs": data_dirs,
                    "split": "dev",
                    "modes": modes
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split('test'),
                gen_kwargs={
                    "data_dirs": data_dirs,
                    "split": "test",
                    "modes": modes
                },
            ),
        ]

    def _generate_examples(self, data_dirs: list, split: str, modes: list):
        all_data = []
        for data_dir in data_dirs:
            # Get all object directories (numbered folders)
            if not os.path.exists(data_dir):
                continue
                
            try:
                object_dirs = [d for d in os.listdir(data_dir) 
                              if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()]
                object_dirs.sort(key=int)  # Sort numerically
            except Exception as e:
                print(f"Error listing directory {data_dir}: {e}")
                continue
            
            # Split data: 80% train, 10% dev, 10% test
            total_objects = len(object_dirs)
            train_num = int(total_objects * 0.8)
            dev_num = int(total_objects * 0.9)
            
            if split == 'train':
                split_objects = object_dirs[:train_num]
            elif split == 'dev':
                split_objects = object_dirs[train_num:dev_num]
            else:  # test
                split_objects = object_dirs[dev_num:]
            
            for obj_dir in split_objects:
                obj_path = os.path.join(data_dir, obj_dir)
                
                # Load goal instruction
                goal_file = os.path.join(obj_path, "goal_instruction.json")
                if not os.path.exists(goal_file):
                    continue
                    
                with open(goal_file, 'r') as f:
                    goal_data = json.load(f)
                    goal_instruction = goal_data.get("goal_instruction", "")
                
                # Load COT descriptions
                cot_file = os.path.join(obj_path, "cot_descriptions.json")
                if not os.path.exists(cot_file):
                    continue
                    
                with open(cot_file, 'r') as f:
                    cot_data = json.load(f)
                    steps = cot_data.get("steps", {})
                
                # Sort steps by step number
                step_numbers = sorted([int(k) for k in steps.keys()])
                
                # Collect step data
                step_list = []
                for step_num in step_numbers:
                    step_key = str(step_num)
                    step_info = steps[step_key]
                    
                    # Get step image
                    step_img_path = os.path.join(obj_path, f"step_{step_num:03d}.png")
                    if not os.path.exists(step_img_path):
                        continue
                    
                    step_list.append({
                        "step": step_num,
                        "description": step_info.get("description", ""),
                        "image_path": step_img_path,
                    })
                
                if len(step_list) == 0:
                    continue
                
                all_data.append({
                    "object_id": obj_dir,
                    "goal_instruction": goal_instruction,
                    "steps": step_list,
                    "task": "3d_assembly"
                })
        
        # Generate interleaved examples
        data_idx = 0
        for data_item in all_data:
            interleaved_data_list = get_interleaved_data(
                data_item,
                mode=modes
            )
            
            for item in interleaved_data_list:
                return_info = {
                    'idx': data_idx,
                    "input_text": item['input_text'],
                    "input_imgs": item["input_imgs"],
                    "label_text": item['label_text'],
                    "label_imgs": item['label_imgs'],
                    "label_img_paths": item['label_img_paths'],
                    "input_img_paths": item['input_img_paths'],
                    "task": item['task'],
                    "train_task": item['train_task'],
                }
                yield data_idx, return_info
                data_idx += 1


def get_interleaved_data(data_item, mode=["single_step_visualization"]):
    """
    Generate interleaved text-image sequences for V3D-CoT.
    
    Format: Goal instruction + (step description, step image) pairs
    For each step t, create an example:
    - Input: goal + (z_1, v_1), (z_2, v_2), ..., (z_{t-1}, v_{t-1})
    - Label: (z_t, v_t)
    """
    interleaved_data = []
    
    goal_instruction = data_item['goal_instruction']
    steps = data_item['steps']
    task = data_item['task']
    
    # Load all step images
    all_step_images = []
    all_step_descriptions = []
    all_image_paths = []
    
    for step in steps:
        img_path = step['image_path']
        try:
            img = Image.open(img_path).convert("RGB").resize((256, 256))
            all_step_images.append(img)
            all_step_descriptions.append(step['description'])
            all_image_paths.append(img_path)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return interleaved_data  # Skip this object if images can't be loaded
    
    if len(all_step_images) == 0:
        return interleaved_data
    
    if "single_step_visualization" in mode:
        # For each step, create a training example
        # Input: goal + previous steps (description + image)
        # Label: current step (description + image)
        
        for step_idx in range(len(steps)):
            # Build input: goal + history of previous steps
            input_text_parts = [f"Goal: {goal_instruction}\n\nResponse: "]
            input_images = []
            input_image_paths = []
            
            # Add previous steps
            for prev_idx in range(step_idx):
                desc = all_step_descriptions[prev_idx]
                input_text_parts.append(desc)
                input_text_parts.append("<image>")
                input_images.append(all_step_images[prev_idx])
                input_image_paths.append(all_image_paths[prev_idx])
            
            input_text = "".join(input_text_parts)
            
            # Label: current step description + image
            label_text = all_step_descriptions[step_idx] + " <image>"
            label_images = [all_step_images[step_idx]]
            label_image_paths = [all_image_paths[step_idx]]
            
            return_info = {
                "task": task,
                "input_text": input_text,
                "label_text": label_text,
                "input_imgs": input_images,
                "input_img_paths": input_image_paths,
                "label_imgs": label_images,
                "label_img_paths": label_image_paths,
                'train_task': "single_step_visualization",
            }
            interleaved_data.append(return_info)
    
    return interleaved_data

