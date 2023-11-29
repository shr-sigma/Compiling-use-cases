# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest
import pickle
import torch
import torch.nn as nn
import os

import shark_turbine.aot as aot

current_directory = os.getcwd()
file_path = os.path.join(current_directory, 'HifiGAN_usecase.pt')
with open(file_path, 'rb') as file:
    modelUseCase = pickle.load(file)
print(modelUseCase)

example_x = torch.ones(1, 1)
exported = aot.export(modelUseCase, example_x)
exported.print_readable()
compiled_binary = exported.compile(save_to=None)


def infer():
    import numpy as np
    import iree.runtime as rt

    config = rt.Config("local-task")
    vmm = rt.load_vm_module(
        rt.VmModule.wrap_buffer(config.vm_instance, compiled_binary.map_memory()),
        config,
    )
    x = np.random.rand(97, 8).astype(np.float32)
    y = vmm.main(x)
    print(y.to_host())


class ModelTest(unittest.TestCase):
    def testMLPExportSimple(selfs):
        infer()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
