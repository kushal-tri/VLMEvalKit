from absl import app, flags
import argparse
import json
import time
import os
import subprocess
import yaml
from datetime import datetime
from pathlib import Path

import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace
from pprint import pprint


NAME = "VLMEvalKit"
INSTANCE_MAPPER = {
    "p3": "ml.p3.16xlarge",
    "p4": "ml.p4d.24xlarge",
    "p4de": "ml.p4de.24xlarge",
    "p5": "ml.p5.48xlarge",
}

SAGE_CONFIG = "sagemaker/yaml_configs/sagemaker.yaml"

def add_flag(name, type, default):
    if type == str:
        flags.DEFINE_string(name=name, default=default, help=f"{name} for the experiment")
    elif type == int:
        flags.DEFINE_integer(name=name, default=default, help=f"{name} for the experiment")
    elif type == float:
        flags.DEFINE_float(name=name, default=default, help=f"{name} for the experiment")
    elif type == bool:
        flags.DEFINE_boolean(name=name, default=default, help=f"{name} for the experiment")
    else:
        raise ValueError(f"Unknown type: {type}")

FLAGS = flags.FLAGS
def setup_flags():
    # read the experiment config and add the flags
    all_keys = []
    with open(EXP_CONFIG, "r") as f:
        exp_config = yaml.safe_load(f)
    for key, value in exp_config.items():
        add_flag(key, type(value), value)
        all_keys.append(key)
    
    # read the sagemaker config and add the flags
    with open(SAGE_CONFIG, "r") as f:
        sagemaker_config = yaml.safe_load(f)
    for key, value in sagemaker_config.items():
        add_flag(key, type(value), value)
        all_keys.append(key)
        
    print("All keys: ", all_keys)
setup_flags()

def get_wandb_run_name(exp_type, FLAGS):
    now = datetime.now()
    # Format example: 2023-03-03-10-14-02-324
    now_ms_str = f"{now.microsecond // 1000:03d}"
    date_str = f"{now.strftime('%Y-%m-%d-%H-%M-%S')}-{now_ms_str}"
    if exp_type == 'rm':
        model_for_run_name = FLAGS.model_name.split('/')[-1]
        run_name = f'prm_lr{FLAGS.lr}_model{model_for_run_name}_dataset{FLAGS.dataset}_schedule{FLAGS.schedule}_{date_str}'
    elif exp_type == 'value':
        model_for_run_name = FLAGS.model_name.split('/')[-1]
        run_name = f"value_lr{FLAGS.lr}_model{model_for_run_name}_disc{FLAGS.discount}_update{FLAGS.update}_obj{FLAGS.objective}_{date_str}"
    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")
        
    print(f"Wandb run name: {run_name}")
    return run_name

def run_command(command):
    print(f"=> {command}")
    subprocess.run(command, shell=True, check=True)


def get_image(user, instance_type, build_type=None, profile="Robotics-LBM-PowerUserAccess-682769330988", region="us-east-1"):
    os.environ["AWS_PROFILE"] = f"{profile}"
    account = subprocess.getoutput(
        f"aws --region {region} --profile {profile} sts get-caller-identity --query Account --output text"
    )
    docker_dir = Path(__file__).parent
    if instance_type in ("p4", "p4de"):
        algorithm_name = f"{user}-{NAME}-p4"
        dockerfile_base = docker_dir / "Dockerfile"
        dockerfile_update = docker_dir / "Dockerfile_update"
    elif instance_type == "p5":
        algorithm_name = f"{user}-{NAME}-p5"
        dockerfile_base = docker_dir / "Dockerfile"
        dockerfile_update = docker_dir / "Dockerfile_update"
    else:
        raise ValueError(f"Unknown instance_type: {instance_type}")
    fullname = f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"
    if build_type is None:
        return fullname

    login_cmd = f"aws ecr get-login-password --region {region} --profile {profile} | docker login --username AWS --password-stdin"

    if build_type == "full":
        print("Building container")
        commands = [
            # Log in to Sagemaker account to get image.
            f"{login_cmd} 763104351884.dkr.ecr.{region}.amazonaws.com",
            f"docker build --progress=plain -f {dockerfile_base} --build-arg AWS_REGION={region} -t {algorithm_name} .",
            f"docker tag {algorithm_name} {fullname}",
            f"{login_cmd} {fullname}",
            (
                f"aws --region {region} ecr describe-repositories --repository-names {algorithm_name} || "
                f"aws --region {region} ecr create-repository --repository-name {algorithm_name}"
            ),
        ]
    elif build_type == "update":
        print("Updating container")
        commands = [
            f"docker build --progress=plain -f {dockerfile_update} --build-arg BASE_DOCKER={algorithm_name} -t {algorithm_name} .",
            f"docker tag {algorithm_name} {fullname}",
            f"{login_cmd} {fullname}",
        ]
    else:
        raise ValueError(f"Unknown build_type: {build_type}")

    # Create command, making sure to exit if any part breaks.
    command = "\n".join([f"{x} || exit 1" for x in commands])
    run_command(command)
    run_command(f"docker push {fullname}")
    print("Sleeping for 5 seconds to ensure push succeeded")
    time.sleep(5)
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"


def main(argv):
    if FLAGS.arn is None:
        assert "SAGEMAKER_ARN" in os.environ, "Please specify --arn or set the SAGEMAKER_ARN environment variable"
        FLAGS.arn = os.environ["SAGEMAKER_ARN"]

    if FLAGS.s3_remote_sync is None:
        assert (
            "S3_REMOTE_SYNC" in os.environ
        ), "Please specify --s3-remote-sync or set the S3_REMOTE_SYNC environment variable"
        FLAGS.s3_remote_sync = os.environ["S3_REMOTE_SYNC"]

    if EXP_TYPE in ['rm', 'value']:
        FLAGS.wandb_run_name = get_wandb_run_name(EXP_TYPE, FLAGS)

    image = get_image(
        FLAGS.user,
        FLAGS.instance_type,
        region=FLAGS.region,
        build_type=FLAGS.build_type,
        profile=FLAGS.profile,
    )

    ##########
    # Create session and make sure of account and region
    ##########
    sagemaker_session = sagemaker.Session(boto_session=boto3.session.Session(region_name=FLAGS.region))

    if FLAGS.local:
        from sagemaker.local import LocalSession
        sagemaker_session = LocalSession()

    role = FLAGS.arn
    # provide a pre-existing role ARN as an alternative to creating a new role
    role_name = role.split(["/"][-1])
    print(f"SageMaker Execution Role:{role}")
    print(f"The name of the Execution role: {role_name[-1]}")

    client = boto3.client("sts")
    account = client.get_caller_identity()["Account"]
    print(f"AWS account:{account}")

    session = boto3.session.Session()
    region = session.region_name
    print(f"AWS region:{region}")

    ##########
    # Configure the training
    ##########
    base_job_name = f"{FLAGS.user.replace('.', '-')}-{NAME}"

    checkpoint_local_path = "/opt/ml/checkpoints"

    def get_job_name(base):
        now = datetime.now()
        # Format example: 2023-03-03-10-14-02-324
        now_ms_str = f"{now.microsecond // 1000:03d}"
        date_str = f"{now.strftime('%Y-%m-%d-%H-%M-%S')}-{now_ms_str}"

        job_name = "_".join([base, date_str])

        return job_name

    job_name = get_job_name(base_job_name)

    output_root = f"{FLAGS.s3_remote_sync}/sagemaker/{FLAGS.user}/{NAME}/"
    output_s3 = os.path.join(output_root, job_name)

    # use yaml to configure the hyperparameters
    hyperparameters = {}
    with open(EXP_CONFIG, "r") as f:
        exp_config = yaml.safe_load(f)
        for key in exp_config.keys():
            value = getattr(FLAGS, key)
            hyperparameters[key] = value
    print("Hyperparameters: ")
    pprint(hyperparameters)

    # TODO: verify if this breaks the code
    environment = {
        "SM_USE_RESERVED_CAPACITY": "1"
    }

    if EXP_TYPE == 'rm':
        entry_point = "qlearning_reasoning/training/rm.py"
    elif EXP_TYPE == 'value':
        entry_point = "qlearning_reasoning/training/value.py"
    elif EXP_TYPE == 'sft':
        entry_point = "qlearning_reasoning/training/sft.py"
    else:
        raise ValueError(f"Unknown experiment type: {EXP_TYPE}")

    estimator = HuggingFace(
        entry_point=entry_point,
        py_version='py310',
        sagemaker_session=sagemaker_session,
        base_job_name=base_job_name,
        hyperparameters=hyperparameters,
        role=role,
        image_uri=image,
        instance_count=FLAGS.instance_count,
        instance_type="local_gpu" if FLAGS.local else INSTANCE_MAPPER[FLAGS.instance_type],
        train_use_spot_instances=FLAGS.spot_instance,
        output_path=output_s3,
        job_name=job_name,
        checkpoint_s3_uri=None if FLAGS.local else f"{output_s3}/checkpoint",
        checkpoint_local_path=None if FLAGS.local else checkpoint_local_path,
        code_location=output_s3,
        # Max run 5 days
        max_run=5 * 24 * 60 * 60,
        max_wait=5 * 24 * 60 * 60 if FLAGS.spot_instance else None,
        input_mode="FastFile",
        environment=environment,
        keep_alive_period_in_seconds=30 * 60 if not FLAGS.spot_instance else None,  # 30 minutes
        tags=[
            {"Key": "tri.project", "Value": "MM:PJ-0077"},
            {"Key": "tri.owner.email", "Value": f"{FLAGS.user}@tri.global"},
        ],
        command=[
            'accelerate', 'launch', '--config-file', 
            '/opt/ml/code/qlearning_reasoning/qlearning_reasoning/accelerate_configs/8gpu.yaml'
        ],
    )
    estimator.fit()


if __name__ == "__main__":
    app.run(main)