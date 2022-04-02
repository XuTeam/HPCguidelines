import torch
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

print('If assigned GPU:', torch.cuda.is_available(), '  Current folder name:', dir_path)