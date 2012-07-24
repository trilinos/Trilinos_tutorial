#!/bin/tcsh

set ROOT = `pwd` 
set pack_list=(advanced beginner custom gui)

cd $ROOT/logs
rm -f *.log

cd $ROOT
rm -f *~
foreach pack ($pack_list)
	cd $ROOT
	set package = $pack
	#######################################################
	set file_name="$package"_list
	ls $package/ >"$file_name"
	set exec_array=(`cat "$file_name"`) 
	rm -f "$package"_list

	cd $package
	foreach exe ($exec_array)
		if(-d $exe) then
		    cd $exe
		    rm -f *.log
		    rm -f *.o
		    rm -f *~
		    rm -f "$exe"
		    rm -rf exe
		    cd ..
		endif
	end
	#######################################################
end
