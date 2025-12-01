from PIL import Image
import torch

from .base import BaseModel
from ..smp import *

from io import BytesIO
import base64
from mimetypes import guess_type


lbm_home = os.environ.get("LBM_HOME", None)
assert lbm_home is not None, "You must set the LBM_HOME environment variable to the path of the LBM diffusion-policy codebase."
if lbm_home not in sys.path:
    sys.path.insert(0, lbm_home)

from diffusion_policy.model.vision.paligemma.paligemma2 import PaliGemma2Model


class PaliGemma2(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='google/paligemma-3b-mix-448', **kwargs):
        try:
            from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
        except Exception as e:
            logging.critical('Please install the latest version transformers.')
            raise e

        model = PaliGemma2Model().cuda().to(torch.bfloat16)

        self.model = model.cuda()
        self.processor = PaliGemmaProcessor.from_pretrained("google/paligemma2-3b-pt-224")
        self.kwargs = kwargs

    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image_path).convert('RGB')

        model_inputs = self.processor(
            text=[prompt], images=[[image],], return_tensors='pt'
        ).to('cuda')
        input_len = model_inputs['input_ids'].shape[-1]

        if type(self.model) == PaliGemma2Model:
            model_inputs['input_ids'] = model_inputs['input_ids'][:, 256:].cuda()
            model_inputs['pixel_values'] = model_inputs['pixel_values'].cuda()

            model_inputs['prefix_masks'] = model_inputs['attention_mask'].cuda()
            del model_inputs['attention_mask']
            with torch.inference_mode():
                generation = self.model.generate(**model_inputs, kv_cache=([], []), do_sample=False)
                res = self.processor.decode(generation[0], skip_special_tokens=True)
                print(res)
        return res