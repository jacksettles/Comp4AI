#!/bin/bash

export PROJHOME=/lustre/orion/csc143/proj-shared/jwang/Comp4AI
cd $PROJHOME/ExternalDependencies/SZ3

git checkout v3.1.8

mkdir -p install

mkdir -p build && cd build

cmake -DCMAKE_INSTALL_PREFIX:PATH=$PROJHOME/ExternalDependencies/SZ3/install ..

make && make install
