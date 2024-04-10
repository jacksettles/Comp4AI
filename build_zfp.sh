#!/bin/bash

export PROJHOME=/lustre/orion/csc143/proj-shared/jwang/Comp4AI

cd $PROJHOME/ExternalDependencies

#git clone git@github.com:LLNL/zfp.git

cd zfp 

git checkout 0.5.5

mkdir -p install

mkdir -p build && cd build

cmake -DCMAKE_INSTALL_PREFIX:PATH=$PROJHOME/ExternalDependencies/zfp/install -DZFP_WITH_OPENMP=OFF -DBUILD_TESTING=OFF \
	-DNumPy_INCLUDE_DIR=/sw/andes/python/3.7/anaconda-base/pkgs/numpy-base-1.18.1-py37hde5b4d6_1/lib/python3.7/site-packages/numpy/core/include \
      -DBUILD_ZFPY=ON -DPYTHON_LIBRARY=$PYTHON_PATH/lib/libpython3.7m.so -DPYTHON_INCLUDE_DIR=$PYTHON_PATH/include/python3.7m ..

make && make install
