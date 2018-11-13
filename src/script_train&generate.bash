declare -a arr=(0 1 2 3 4 5 6 7 8 9)
declare -a im=(5,200 10,200 25,250 50,300 100,350)
declare -a latent=(2 3 4 8)

for lat in "${latent[@]}"
do
	for i in "${arr[@]}"
	do
		for j in "${im[@]}"
			do
				IFS=","
				set $j
				printf "\n\nTRAINING $i VAE USING $1 IMAGES\n\n"
				python VAE.py $i $1 $2 $lat
			done
	done

	for i in "${arr[@]}"
	do
		for j in "${im[@]}"
			do
				IFS=","
				set $j
				printf "\n\nGENERATING $i SAMPLES USING $1 IMAGES\n\n"
				python GENERATOR.py $i $1 $2 $lat
			done
	done
done

