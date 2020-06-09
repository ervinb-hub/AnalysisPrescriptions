#!/bin/bash

#default parameters
DATA="sample"
VERBOSE="no"
SPARK="spark-submit"
VERSION=1

#cd /home/ebeta001/test_big_data_project

source activate py27 > /dev/null 

function usage() {
    echo "Usage: ./run.sh [--data='sample'|'full'] [--verbose='yes'|'no'] [--version=1|2]"
    echo
    echo "--data       Specify the quantity of data to process: sample=~3GB. Default: sample"
    echo "--verbose    Specify the level of verbosity. If true logs are shown on screen, otherwise saved on out.log. Default: no"
    echo "--version    Specify the version of pyspark. Default: 1 (1.4.0)"
}

function parse_arguments() {
    while [ "$1" != "" ]; do
        PARAM=`echo $1 | awk -F= '{print $1}'`
        VALUE=`echo $1 | awk -F= '{print $2}'`
        case $PARAM in
            -h | --help)
                usage
                exit
                ;;
            --data)
                DATA=$VALUE
                ;;
            --verbose)
                VERBOSE=$VALUE
                ;;
	    --version)
		VERSION=$VALUE
		;;
            *)
                echo "ERROR: Incorrect use of parameter \"$PARAM\""
                usage
                exit 1
                ;;
        esac
        shift
    done
}

parse_arguments $@

if [ $DATA == "full" ]; then
    hadoop fs -rm -f * > /dev/null 
    for file in $(ls ../full_data/*.CSV); do
	echo "Copying $file to HDFS"
	hadoop fs -copyFromLocal ../full_data/$file
    done
elif [ "$DATA" == "sample" ]; then
    hadoop fs -rm -f * /dev/null
    for file in $(ls *.CSV); do
	echo "Copying $file to HDFS"
	hadoop fs -copyFromLocal $file
    done
else
    echo "ERROR: --data can be 'sample' or 'full'"
    usage
fi


if [ "$VERBOSE" == "yes" ]; then
    VERBOSE=""
elif [ "$VERBOSE" == "no" ]; then
    VERBOSE="2>out.log"
else
     echo "ERROR: --version can be 'yes' or 'no'"
     usage
fi


if [[ $VERSION -eq 2 ]]; then
    SPARK="spark-submit-21"
elif [[ $VERSION -eq 1 ]]; then
    SPARK="spark-submit"
else
    echo "ERROR: --version can be 1 or 2"
    usage
fi

echo "Running: nohup $SPARK --master='yarn' final_draft.py 1> output $VERBOSE"
nohup $SPARK --master='yarn' final_draft.py 1> output $VERBOSE

