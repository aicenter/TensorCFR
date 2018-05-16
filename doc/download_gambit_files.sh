#!/usr/bin/env bash
download_gambit_files () {
    # load filenames according to the domain $arg
    command=". $1/download_gambit_files.sh"
    echo -e "\n$command"
    ${command}

    # download domain files
    for file in "${files[@]}";
    do
            cd $1
            url="https://owncloud.cesnet.cz/index.php/s/$file/download"
            wget --no-check-certificate --content-disposition -q --show-progress $url
            ls | grep ".gz" | xargs gzip -f -d
            cd ..
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
		download_gambit_files $PWD/$arg
	done
fi
