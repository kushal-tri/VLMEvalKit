from PIL import Image
import torch
from transformers import AutoProcessor


from .base import BaseModel
from ..smp import *

import boto3

from io import BytesIO
import base64
from mimetypes import guess_type
from types import SimpleNamespace

def deferred_imports():
    # TODO(imcmahon): Remove these deferred imports. This was only useful
    # for how the Anzu code was structured (e.g., don't pay for cost of
    # importing diffusion_policy if you're only doing teleop).

    # These are very slow, and if we're doing teleop or another analysis, we
    # don't want to pay this import cost.
    import hydra
    from omegaconf import OmegaConf

    # Transitively imported later; fail-fast here.
    import transformers  # noqa: F401

    import diffusion_policy.common.gdown_and_cache_dir_mods  # noqa: F401
    from diffusion_policy.common.pose_util import pose9d_to_mat
    from diffusion_policy.common.pytorch_util import dict_apply
    from diffusion_policy.dataset.relative_trajectory_conversion import (
        change_rel_action_to_abs,
        change_to_relative_trajectories,
    )
    from diffusion_policy.model.common.normalizer import (
        SingleFieldLinearNormalizer,
    )
    from diffusion_policy.model.common.rotation_transformer import (
        RotationTransformer,
    )
    from diffusion_policy.policy_wrapper.transform import pretrained_normalize

    # N.B. pycodestyle + isort + black fight here. Just let them have their
    # way.
    from diffusion_policy.workspace.lightning_workspace import (
        TrainDiffusionLightningWorkspace,
    )

    return SimpleNamespace(
        dict_apply=dict_apply,
        hydra=hydra,
        pretrained_normalize=pretrained_normalize,
        OmegaConf=OmegaConf,
        RotationTransformer=RotationTransformer,
        SingleFieldLinearNormalizer=SingleFieldLinearNormalizer,
        TrainDiffusionLightningWorkspace=TrainDiffusionLightningWorkspace,
        pose9d_to_mat=pose9d_to_mat,
        change_to_relative_trajectories=change_to_relative_trajectories,
        change_rel_action_to_abs=change_rel_action_to_abs,
    )


class LBM1(BaseModel):
    def __init__(self, checkpoint_path=None, **kwargs):
        super().__init__(**kwargs)
        lbm_home = os.environ.get("LBM_HOME", None)
        assert lbm_home is not None, "You must set the LBM_HOME environment variable to the path of the LBM diffusion-policy codebase."
        if lbm_home not in sys.path:
            sys.path.insert(0, lbm_home)


        from diffusion_policy.aws.s3_util import (
            get_aws_region_from_s3_path,
            is_s3_path,
            list_s3_files_with_boto3,
            maybe_download_from_s3,
        )

        from diffusion_policy.common.path_util import resolve_path

        # Check if checkpoint_path is specified; if not, fall back to env variable.
        lbm_ckpt_env = os.environ.get("LBM_CHECKPOINT_PATH", None)
        if lbm_ckpt_env is not None:
            checkpoint_path = lbm_ckpt_env
            print(f"Loading LBM1 model from environment variable LBM_CHECKPOINT_PATH.")
        elif checkpoint_path is None:
            raise ValueError(
                "You must specify a checkpoint_path argument, or set LBM_CHECKPOINT_PATH environment variable."
            )

        print(f"Loading LBM1 model from: {checkpoint_path}")

        # Assuming ckpt_path is a directory, list the files inside,
        # confirm there is only one ".ckpt" file and return it.
        if checkpoint_path.endswith(".ckpt"):
            ckpt_list = [checkpoint_path]
        elif is_s3_path(checkpoint_path):
            region = get_aws_region_from_s3_path(checkpoint_path)
            client = boto3.client("s3", region_name=region)
            files = list_s3_files_with_boto3(
                s3_client=client, s3_dir=checkpoint_path, recursive=False
            )
            client.close()
            ckpt_list = [f for f in files if f.endswith(".ckpt")]
        else:
            raise ValueError(f"Invalid checkpoint file: {checkpoint_path}")

        if len(ckpt_list) != 1:
            raise ValueError(f"Expected 1 checkpoint file, got {len(ckpt_list)}")
        checkpoint_path = ckpt_list[0]

        checkpoint_file = maybe_download_from_s3(checkpoint_path)

        workspace_cls = deferred_imports().TrainDiffusionLightningWorkspace
        workspace = workspace_cls.create_from_checkpoint(
            checkpoint_path=resolve_path(checkpoint_file),
            output_dir="/tmp",  # Required argument, unimportant for inference.
            map_location=torch.device("cpu"),
        )
    
        vla_model =  workspace.lightning_module_wrapper.model
        self.model = vla_model.vlm.cuda().eval()


        base_vlm = workspace.resolved_cfg.tokenizer.base_vlm
        self.processor = AutoProcessor.from_pretrained(base_vlm)

    def generate_inner(self, message, dataset=None):     
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image = Image.open(image_path).convert('RGB')

        model_inputs = self.processor(
            text=[prompt], images=[[image],], return_tensors='pt'
        ).to('cuda')
        input_len = model_inputs['input_ids'].shape[-1]

        from diffusion_policy.model.vision.paligemma.paligemma2 import PaliGemma2Model
        if type(self.model) == PaliGemma2Model:
            model_inputs['input_ids'] = model_inputs['input_ids'][:, 256:].cuda()
            model_inputs['pixel_values'] = model_inputs['pixel_values'].cuda()

            model_inputs['prefix_masks'] = model_inputs['attention_mask'].cuda()
            del model_inputs['attention_mask']
            with torch.inference_mode():
                generation = self.model.generate(**model_inputs, kv_cache=([], []))
                res = self.processor.decode(generation[0], skip_special_tokens=True)

        return res