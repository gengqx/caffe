#!/usr/bin/env bash
mkdir build
cd build 
cmake -DCUDA_ARCH_NAME=All -DBUILD_docs=OFF -DBUILD_python_layer=OFF -DUSE_OPENCV=OFF -DUSE_LEVELDB=OFF -DUSE_LMDB=OFF ..
make all -j 10
make install

cd ..
mkdir caffe-output
cp -r build/install/include caffe-output
cp -r build/lib caffe-output
