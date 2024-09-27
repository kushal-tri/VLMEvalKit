import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json

import boto3
import draccus
import sagemaker
from sagemaker.pytorch import PyTorch

from prismatic.conf import DatasetRegistry, ModelRegistry

NAME = "vlmevalkit"
INSTANCE_MAPPER = {
    "p4": "ml.p4d.24xlarge",
    "p4de": "ml.p4de.24xlarge",
    "p5": "ml.p5.48xlarge",
    "g5": "ml.g5.48xlarge"
}


@dataclass
class LaunchConfig:
    model_path: str = None
    data: str = "MMBench_DEV_EN"
    model: str = "prismatic"
    
    # Sagemaker configs.
    local: bool = False
    user: str = None

    # AWS profile args
    region: str = "us-east-1"
    profile: str = "default"
    arn: str = None
    s3_remote_sync: str = "s3://tri-ml-datasets/mbm/VLMEvalKit/"

    # Instance args
    instance_count: int = 1
    instance_type: str = "g5"
    pool_capacity: bool = False
    spot_instance: bool = False

def run_command(command):
    print(f"=> {command}")
    subprocess.run(command, shell=True, check=True)

def get_image(user, instance_type, profile="default", region="us-east-1"):
    os.environ["AWS_PROFILE"] = f"{profile}"
    account = subprocess.getoutput(
        f"aws --region {region} --profile {profile} sts get-caller-identity --query Account --output text"
    )
    docker_dir = Path(__file__).parent
    if instance_type in ("p4", "p4de", "p5", "g5"):
        algorithm_name = f"{user}-{NAME}"
        dockerfile_base = docker_dir / "Dockerfile"
        dockerfile_update = docker_dir / "Dockerfile_update"
    else:
        raise ValueError(f"Unknown instance_type: {instance_type}")
    fullname = f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"

    login_cmd = (
        f"aws ecr get-login-password --region {region} --profile {profile} |     docker login --username AWS"
        " --password-stdin"
    )

    print("Building container")
    commands = [
        # Log in to Sagemaker account to get image.
        f"{login_cmd} 763104351884.dkr.ecr.{region}.amazonaws.com",
        f"docker build -f {dockerfile_base} --build-arg AWS_REGION={region} -t {algorithm_name} .",
        f"docker tag {algorithm_name} {fullname}",
        f"{login_cmd} {fullname}",
        (
            f"aws --region {region} ecr describe-repositories --repository-names {algorithm_name} || "
            f"aws --region {region} ecr create-repository --repository-name {algorithm_name}"
        ),
    ]

    # Create command, making sure to exit if any part breaks.
    command = "\n".join([f"{x} || exit 1" for x in commands])
    run_command(command)
    run_command(f"docker push {fullname}")
    print("Sleeping for 5 seconds to ensure push succeeded")
    time.sleep(5)
    return f"{account}.dkr.ecr.{region}.amazonaws.com/{algorithm_name}:latest"


@draccus.wrap()
def main(args: LaunchConfig):
    assert args.model_path is not None, "Please specify model_path to evaluate."
    assert args.model_path.startswith("s3://"), "model_path needs to an s3 repo."

    if args.arn is None:
        assert "SAGEMAKER_ARN" in os.environ, "Please specify --arn or set the SAGEMAKER_ARN environment variable"
        args.arn = os.environ["SAGEMAKER_ARN"]

    image = get_image(
        args.user,
        args.instance_type,
        region=args.region,
        profile=args.profile,
    )

    ##########
    # Create session and make sure of account and region
    ##########
    sagemaker_session = sagemaker.Session(boto_session=boto3.session.Session(region_name=args.region))

    if args.local:
        from sagemaker.local import LocalSession

        sagemaker_session = LocalSession()

    role = args.arn
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
    base_job_name = f"{args.user.replace('.', '-')}-{NAME}"

    checkpoint_local_path = "/opt/ml/checkpoints"

    def get_job_name(base):
        now = datetime.now()
        # Format example: 2023-03-03-10-14-02-324
        now_ms_str = f"{now.microsecond // 1000:03d}"
        date_str = f"{now.strftime('%Y-%m-%d-%H-%M-%S')}-{now_ms_str}"

        job_name = "_".join([base, date_str])

        return job_name

    job_name = get_job_name(base_job_name)

    if args.model_path[-1] == "/":
        args.model_path = args.model_path[:-1]

    splits = args.model_path.split("/")
    folder_name, filename = splits[-2], splits[-1]

    output_root = f"{args.s3_remote_sync}/sagemaker/{args.user}/{NAME}/"
    output_s3 = os.path.join(output_root, folder_name, filename, job_name)

    # Compute Batch Size Parameters
    world_size = args.instance_count * 8
    if args.per_device_batch_size is None:
        args.per_device_batch_size = args.global_batch_size // world_size
    assert args.global_batch_size % world_size == 0, f"World Size `{world_size}` does not divide global batch size!"
    

    hyperparameters = {
        "model": args.model,
        "data": args.data,
        "model_path": args.model_path,
        "work-dir": "/opt/ml/output/data"
    }

    estimator = PyTorch(
        entry_point="run.py",
        sagemaker_session=sagemaker_session,
        base_job_name=base_job_name,
        hyperparameters=hyperparameters,
        role=role,
        image_uri=image,
        instance_count=args.instance_count,
        instance_type="local_gpu" if args.local else INSTANCE_MAPPER[args.instance_type],
        train_use_spot_instances=args.spot_instance,
        output_path=output_s3,
        job_name=job_name,
        checkpoint_s3_uri=None if args.local else f"{output_s3}/checkpoint",
        checkpoint_local_path=None if args.local else checkpoint_local_path,
        code_location=output_s3,
        # Training using SMDataParallel Distributed Training Framework
        distribution={"torch_distributed": {"enabled": True}},
        # Max run 5 days
        max_run=5 * 24 * 60 * 60,
        max_wait=5 * 24 * 60 * 60 if args.spot_instance else None,
        input_mode="FastFile",
        # environment={"TORCH_DISTRIBUTED_DEBUG": "DETAIL", "TORCH_CPP_LOG_LEVEL": "INFO"},
        environment={"SM_USE_RESERVED_CAPACITY": "0" if args.pool_capacity else "1", "NCCL_DEBUG": "INFO"},
        keep_alive_period_in_seconds=30 * 60 if not args.spot_instance else None,  # 30 minutes
        tags=[
            {"Key": "tri.project", "Value": "MM:PJ-0077"},
            {"Key": "tri.owner.email", "Value": f"{args.user}@tri.global"},
        ],
    )

    estimator.fit()


if __name__ == "__main__":
    main()
