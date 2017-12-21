#!/bin/sh -x 
path_out=sample1.txt
rm $path_out.zip
rm $path_out
make clean
make
./demo $path_out
zip $path_out.zip $path_out
