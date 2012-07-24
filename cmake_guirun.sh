#!/bin/tcsh
#
#use if cmake build is made and configured
#
# for use with the guis
#
# input params:
#	$1 = package
#	$2 = example = exename
#	$3 = numProcs
#	$4 = exeargs
#

#load the necessary modules to make and compile the example files
set file_name=modules
set the_modules=(`cat "$file_name"`)
module load $the_modules
module list

set ROOT = `pwd`/cmake_build

cd $ROOT
echo `pwd`

set package="$1"
set exe="$2"
set numProcs="$3"
set exeargs="$4"

# if valid target make target (more or less just to make sure than any real value)
if(-d "$package/$exe") then
#	echo "good"
	cd "$package/$exe"
	if("$package" == "advanced") then
		set logfile="$exe.$exeargs.log"
	else 
		set logfile="$exe.log"
	endif
	mpiexec -n $numProcs $exe $exeargs >& $ROOT/../logs/"$logfile"
	cat $ROOT/../logs/"$logfile"
else
	echo "bad directory"
endif
echo "DONE!"
