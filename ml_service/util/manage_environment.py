
import os
import traceback
from azureml.core import Workspace, Environment
from ml_service.util.env_variables import Env
from azureml.core.runconfig import DEFAULT_CPU_IMAGE, DEFAULT_GPU_IMAGE


def get_environment(
    workspace: Workspace,
    environment_name: str,
    conda_dependencies_file: str,
    create_new: bool = False,
    enable_docker: bool = None,
    use_gpu: bool = False
):
    try:
        e = Env()
        environments = Environment.list(workspace=workspace)
        restored_environment = None
        for env in environments:
            if env == environment_name:
                restored_environment = environments[environment_name]

        if restored_environment is None or create_new:
            new_env = Environment.from_conda_specification(
                environment_name,
                os.path.join(e.sources_directory_train, conda_dependencies_file),  # NOQA: E501
            )  # NOQA: E501
            restored_environment = new_env
            if enable_docker is not None:
                restored_environment.docker.enabled = enable_docker
                restored_environment.docker.base_image = "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest"
            restored_environment.register(workspace)

        if restored_environment is not None:
            print(restored_environment)
        return restored_environment
    except Exception:
        traceback.print_exc()
        exit(1)
