#!/bin/bash

# compile and install the deep learning modules for 3D meshes and point clouds
bash segmesh/mesh/pi_modules/source/compile.sh
bash segmesh/point/pi_modules/source/compile.sh

# after installation, delete the temporary files in the folder
rm -rf build
rm -rf dist
find . -type d -name "*.egg-info" -exec rm -rf {} +