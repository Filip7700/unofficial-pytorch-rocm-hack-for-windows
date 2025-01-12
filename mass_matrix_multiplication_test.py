#-*- coding: utf-8 -*-

import sys
import torch_rocm_win as torch_rocm



def main(args):
    for i in range(100):
        # Create two large random matrices
        matrix_a = torch_rocm.torch.randn(
            (10000, 10000),
            device = torch_rocm.zluda_device)

        matrix_b = torch_rocm.torch.randn(
            (10000, 10000),
            device = torch_rocm.zluda_device)

        # Multiply matrices, calculate the dot product
        result = torch_rocm.torch.matmul(matrix_a, matrix_b)

        print(f"Result {i}:\n{result}")

    return 0



if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
