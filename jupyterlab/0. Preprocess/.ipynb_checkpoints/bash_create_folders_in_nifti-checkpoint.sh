#!/bin/bash
locproj=/Users/nova/Desktop/wmd_2022_choco

locinfo=info
locdata=data
 
for SUB in 17; do
	echo "$locproj/$locdata/sub-$SUB"
	mkdir $locproj/$locdata/sub-$SUB
	
	mkdir $locproj/$locdata/sub-$SUB/anat
	mkdir $locproj/$locdata/sub-$SUB/fmap
	mkdir $locproj/$locdata/sub-$SUB/func

done
