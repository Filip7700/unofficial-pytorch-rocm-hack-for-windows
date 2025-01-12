#-*- coding: utf-8 -*-

import sys
import os
import ctypes



# GLOBALS
GPU_DEVICE_INDEX = 0

HIPSDK_TARGETS = ["rocblas.dll", "rocsolver.dll", "hiprtc0602.dll"]

ZLUDA_TARGETS = ("nvcuda.dll", "nvml.dll",)

DLL_MAPPING = {
    "cublas.dll": "cublas64_11.dll",
    "cusparse.dll": "cusparse64_11.dll",
    "nvrtc.dll": "nvrtc64_112_0.dll"}



def check_rocm_installation():
    ret = False

    hip_path = os.environ.get("HIP_PATH", "")

    if len(hip_path) > 0:
        ret = True

    return ret



def find_zluda_path():
    path_environment_variable = os.environ.get("Path", "")
    path_environment_variables = path_environment_variable.split(";")

    zluda_path = ""

    for path_item in path_environment_variables:
        # Check for extracted ZLUDA folder at the end of the path
        if "zluda" in path_item[-5:]:
            zluda_path = path_item
            break

    return zluda_path



def load_zluda():
    global HIPSDK_TARGETS
    global ZLUDA_TARGETS
    global DLL_MAPPING

    is_zluda_loaded = False

    is_rocm_installed = check_rocm_installation()

    if is_rocm_installed:
        os.environ["ZLUDA_COMGR_LOG_LEVEL"] = "1"

        rocm_path = os.environ.get("HIP_PATH", "")
        zluda_path = find_zluda_path()

        # CUDA DLL hijacking
        if len(rocm_path) > 0:
            if len(zluda_path) > 0:
                for v in HIPSDK_TARGETS:
                    ctypes.windll.LoadLibrary(os.path.join(rocm_path, "bin", v))

                for v in ZLUDA_TARGETS:
                    ctypes.windll.LoadLibrary(os.path.join(zluda_path, v))

                for v in DLL_MAPPING.values():
                    ctypes.windll.LoadLibrary(os.path.join(zluda_path, v))

                is_zluda_loaded = True
            else:
                print("ZLUDA not found in PATH and not loaded!")
        else:
            print(
                "ROCm installation path not be specified in environment "
                "variables. ROCm HIP SDK may not be installed.")
    else:
        print("ROCm not installed!")

    return is_zluda_loaded



is_zluda_loaded = load_zluda()

# Import torch hack
# The reason the torch is not imported at the beginning of program
# is because the ZLUDA and ROCm should load their DLLs first,
# to trick the PyTorch that the CUDA device exists.
# Otherwise, following error will occur:
# CUDA error: CUBLAS_STATUS_NOT_SUPPORTED
# When using ZLUDA just import this Python module, DO NOT import torch
# before this module, the module imports torch correctly.
import torch

zluda_device = torch.device("cpu")

if is_zluda_loaded:
    zluda_device = torch.device("cuda")
    print("ZLUDA device successfully loaded!")
else:
    print("ZLUDA failed to load, falling back to CPU!")
