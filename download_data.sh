#!/bin/sh

suffixes="PDPI+BNFT ADDR+BNFT CHEM+SUBS"

for line in `cat files.txt`; do
	var1=$(echo $line | awk -F ',' '{print $1}')
	var2=$(echo $line | awk -F ',' '{print $2}')

	echo Downloading data for $var1
	
	for suffix in `echo $suffixes`; do
        echo "http://datagov.ic.nhs.uk/presentation/${var1}/T${var2}${suffix}.CSV"
		wget "http://datagov.ic.nhs.uk/presentation/${var1}/T${var2}${suffix}.CSV"
        echo "\n"
		if [ $? -ne 0 ]; then
			du -sh *
			echo "An error occurred whilst downloading the last file"
			exit $?
		fi
	done
done
