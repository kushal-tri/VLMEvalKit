import torch
from PIL import Image
from .base import BaseModel
from ..smp import *
import copy

from prismatic import load


class Prismatic(BaseModel):

    def __init__(self, model_path, **kwargs):
      self.model = load(model_path, **kwargs)
      self.model.cuda().eval()

    def generate_inner(self, message, dataset=None):
      content, images = '', []
      for msg in message:
          if msg['type'] == 'text':
              content += msg['value']
          else:
              images.append(Image.open(msg['value']).convert('RGB'))
      
      return self.model.generate(
        images[0], 
        content,
        do_sample=False, 
        temperature=1.0,
        max_new_tokens=512,
        min_length=1,
      )
