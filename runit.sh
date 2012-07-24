#!/bin/tcsh

#check if special case get help)
if($1 == "" || $1 == "-h") then
	echo "This script is designed to make and execute a given example from the beginner package. it will still be able to work with other targets but due to the inability to easily pass arguments it is not reasonable to work with the advanced examples."
	echo "Parameters:"
	echo ""
	echo "1 = the path to the target executable or package "
	echo "   (though in reality beginner is the only package guarenteed to work correctly)"
	echo ""
	echo "2 = the number of procs"
	echo ""
	echo "3 = the extra arguments. this will assume they've been passed within single quotes('') not just added at the end "
	echo ""
	echo "EX. ./runit.sh beginner/Epetra_Simple_Vector/ 2"
	echo "Executes the Epetra Simple Vector example with 2 threads"
	echo ""
	echo "It is possible to execute advanced examples however for them to run correctly they require their own specific parameters"
	echo ""
	echo "For example, the Epetra Basic Perf Test takes 6 parameters but these can be by the valid runit.sh command:"
	echo "./runit.sh advanced/Epetra_Basic_Perf/ 2 '5 5 2 1 9 -v'"
else
	set var1="$1"
	set var2="$2"

	# assign the commandline arguments to the correct variables
	set target=$var1
	# set np to a valid number. only 0-9 for now. default is 4
	set np=""
	if("$var2" > "0" && "$var2" <= "9") set np="$var2"
	endif
	if ($np == "") set np=4
	endif
	#set exeargs if available
	set exeargs="$3"

	#load the necessary modules to make and compile the example files
	set file_name=modules
	set the_modules=(`cat "$file_name"`)
	module load $the_modules
	module list
	set ROOT = `pwd`

	# test the target is a directory then parse package and example from the filepath
	if(-d $target) then
		echo "good"
		echo $target | awk -F"/" '{print $1,$2,$3}' > "file_path"
		set file_name="file_path"
		set vars=(`cat "$file_name"`) 
		rm -f $file_name
		set package=${vars[1]}

		if(${#vars} > 1) then
			set exec_array=(${vars[2]})
		else 
			set file_name="$package"_list
			ls $package/ >"$file_name"
			set exec_array=(`cat "$file_name"`) 	
			rm -f "$file_name"
		endif

		echo $exec_array
	else
		echo "bad"
	endif

	# for each target example (either 1 or all) make the executable
	cd $package
	foreach exe ($exec_array)
		if(-d $exe)then
		    cd $exe
		    echo "make $exe "
		    make >& make.log
		    cd ..
		endif
	end

	echo `pwd`
	echo " "

	# for each target example (either 1 or all) execute the executable and make a log for the run information
	foreach exe ($exec_array)  
		if(-d $exe)then
		    cd ${exe}
		    set log=${exe}.log
		    mpiexec -n $np ./$exe $exeargs >& $log
		    cat $log
		    echo "see $exe.log for run information"
		    cd ..
		endif    
	end
endif
