#!/bin/bash

ILCSOFT=/home/ilc/ilcsoft/v02-01-pre/

source $ILCSOFT/init_ilcsoft.sh

echo $ILCSOFT

#cd /home/ilc/data/StandardConfig/production

export SIM_MODEL=ILD_l5_v02

##
## Generate G4 showers with gun DDSim
##
echo "-- Running DDSim ${SIM_MODEL} ..."
ddsim \
    --outputFile photon-shower-$1.slcio \
  --compactFile $lcgeo_DIR/ILD/compact/${SIM_MODEL}/${SIM_MODEL}.xml \
  --steeringFile ddsim_steer_gun.py > ${SIM_MODEL}.$1.log  

