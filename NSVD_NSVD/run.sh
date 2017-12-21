#!/bin/sh -x 
path_out=sample2.txt
rm sample1.txt.zip
rm $path_out
make clean
make
./demo $path_out
zip sample1.txt.zip $path_out
