#!/usr/bin/env bash

download_gambit_files () {
    # load filenames according to the domain $arg
    command=". $1/download_gambit_files.sh"
    echo -e "\n$command"
    ${command}

    # download domain files
    for file in "${files[@]}";
    do
        fileraw=(${file//:/ }) # split ':' into spaces and make an array from the result
        filename=${fileraw[0]}
        filecode=${fileraw[1]}

        if [ ! -f $1/$filename ];
        then
            cd $1
                    url="https://owncloud.cesnet.cz/index.php/s/$filecode/download"
                    wget --no-check-certificate --content-disposition -q --show-progress $url
                    ls | grep ".gz" | xargs gzip -f -d
                    cd ..
        else
            echo "$filename already exists (skipping download)"
        fi
    done
}

echo "DOWNLOAD GAMBIT FILES"

if [ $# -eq 0 ];
then
	for filename in $PWD/*;
	do
		if [ -d $filename ];
		then
			if [ -e $filename/download_gambit_files.sh ];
			then
				download_gambit_files $filename
			else
				echo -e "\nskipping $filename - no download_gambit_files.sh"
			fi
		fi
	done
else
    if [ "$1" = "-h" -o "$1" = "--help" ]; then
        echo "Help:"
        echo -e "\nThe script will download Gambit files from Cesnet ownCloud according to '<domain>/download_gambit_files.sh' specified arrays 'files'."
        echo -e "\nIf you run the script with no parameters, if will scan the 'doc/' subdirectories looking for 'download_gambit_files.sh' and downloading all files, that are there specified."
        echo -e "\nAlternatively you can download Gambit files for just one domain. For example 'bash download_gambit_files.sh poker' will look into 'doc/poker/download_gambit_files.sh' and download all files listed there."
        exit 0;
    fi
	for arg;
	do
	    if [ -f $PWD/$arg/download_gambit_files.sh ];
	    then
		    download_gambit_files $PWD/$arg
		else
		    echo -e "\nskipping $filename - no download_gambit_files.sh"
		fi
	done
fi
