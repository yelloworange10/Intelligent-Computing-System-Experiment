#!/bin/bash

set -ex

rm -rf build && mkdir build
cncc --bang-arch=compute_30 -O3 01_scalar_single_core.mlu -o build/01_scalar_single_core && ./build/01_scalar_single_core
cncc --bang-arch=compute_30 -O3 02_vector_single_core.mlu -o build/02_vector_single_core && ./build/02_vector_single_core
cncc --bang-arch=compute_30 -O3 03_vector_single_core_pipeline.mlu -o build/03_vector_single_core_pipeline && ./build/03_vector_single_core_pipeline
cncc --bang-arch=compute_30 -O3 04_vector_multi_core_pipeline.mlu -o build/04_vector_multi_core_pipeline && ./build/04_vector_multi_core_pipeline
