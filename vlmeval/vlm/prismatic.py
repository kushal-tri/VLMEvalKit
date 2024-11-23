import torch
from PIL import Image

from vlmeval.dataset.image_caption import ImageCaptionDataset
from .base import BaseModel
from ..smp import *
import copy

from prismatic import load


class Prismatic(BaseModel):

    def __init__(self, model_path, **kwargs):
      self.model = load(model_path, **kwargs)
      self.model.cuda().eval()
      self.img_cap_obj = None
    def __get_image_cap_obj(self, dataset):
      assert dataset == "COCO_VAL"
      if self.img_cap_obj is None:
        self.img_cap_obj = ImageCaptionDataset(dataset)
      return self.img_cap_obj 

    def build_prompt(self, line, dataset):
      if dataset == "COCO_VAL":
        line.question == "Provide a short image description.<|/h|>"
      return self.__get_image_cap_obj(dataset).build_prompt(line)
      

    def use_custom_prompt(self, dataset):
      if dataset == "COCO_VAL":
        return True # super().use_custom_prompt(dataset)
      return False
  

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
