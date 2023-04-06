#!/bin/bash
for file in $PWD/*.plt
    do
        filename=$(basename "$file")
        /usr/local/tecplot/360ex_2022r1/bin/tec360 "$file" -o "$PWD"/"${filename%.plt}.szplt"
    done
