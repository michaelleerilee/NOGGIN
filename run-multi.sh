#!/bin/bash


lonA=-179
lonB=179
latA=-60
latB=60

# lonA=-165
# lonB=-145

# Tropics
# latA=-30
# latB=30

# Hawaii
# lonA=-165
# lonB=-145
# latA=10
# latB=30

# Near the pole
# latA=-88
# latB=-68

dlon=2
dlat=2

resolution=0.25

for lat0 in `seq $latA $dlat $latB`; do
    for lon0 in `seq $lonA $dlon $lonB`; do
	# echo $dlon $dlat
	(( lon1 = lon0 + dlon ))
	(( lat1 = lat0 + dlat ))
	./run-vnp02.sh $lon0 $lat0 $lon1 $lat1 $resolution
    done
done
