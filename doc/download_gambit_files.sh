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

echo "Downloading domains:"

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
	for arg;
	do
	    echo $arg

	    if [ -d $PWD/$arg/download_gambit_files.sh ];
	    then
		    download_gambit_files $PWD/$arg
		else
		    echo -e "skipping $filename - no download_gambit_files.sh"
		fi
	done
fi
