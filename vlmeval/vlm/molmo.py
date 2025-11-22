import torch
from PIL import Image
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE

TYPE_PROMPTS = {
    'Y/N':'vqa2:',
    'VQA':'vqa2:',
    'MCQ':'a_okvqa_mc:',
}

DATASET_PROMPTS = {
    'AI2D_TEST':'ai2_diagram:',
    'AI2D_TEST_NO_MASK':'ai2_diagram:',
    'COCO_VAL':'coco_captioning:',
    'ChartQA_TEST':'chart_qa:',
    'ChartQA_VAL':'chart_qa:',
    'DocVQA_VAL':'doc_qa:',
    'DocVQA_TEST':'doc_qa:',
    'InfoVQA_TEST':'info_qa:',
    'InfoVQA_VAL':'info_qa:',
    'OCRVQA_TEST':'ocr_vqa:',
    'OCRVQA_TESTCORE':'ocr_vqa:',
    'ScienceQA_VAL':'science_qa:',
    'ScienceQA_TEST':'science_qa:',
    'TableVQABench':'tabwmp_da:',
    'TextVQA_VAL':'text_vqa:'
}


class molmo(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='allenai/Molmo-7B-D-0924', **kwargs):
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
            import einops
        except Exception as e:
            logging.critical('Please install transformer and einops before using molmo.')
            raise e

        if '72b' not in model_path.lower():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map='cuda')
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto")

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.kwargs = kwargs
        self.model_name = model_path

        # process the image and text
        max_crops = self.max_crops
        inputs = self.processor.process(
            images=[image],
            text=prompt,
            images_kwargs={
                "max_crops": max_crops
            }
        )

        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(self.model.device).unsqueeze(0) for k, v in inputs.items()}

        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            output = self.model.generate_from_batch(
                inputs,
                GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                tokenizer=self.processor.tokenizer
            )

        # only get generated tokens; decode them to text
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # AI2D: map direct answer to letter option
        if dataset in ['AI2D_TEST', 'AI2D_TEST_NO_MASK']:
            # 'ai2_diagram_no_letter: Which of the following is the magma chamber?\nK\nB\nC\nH'
            if 'ai2_diagram_no_letter' in prompt:
                options = prompt.split('\n')[1:]
                answer = options.index(generated_text)
                generated_text = chr(answer + ord('A'))

        # print(dataset, prompt, generated_text, inputs['images'].size()) # uncomment to debug

        return generated_text
