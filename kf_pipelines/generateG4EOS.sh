#!/bin/bash

ILCSOFT=/home/ilc/ilcsoft/v02-01-pre/

source $ILCSOFT/init_ilcsoft.sh

echo $ILCSOFT

#cd /home/ilc/data/StandardConfig/production

export SIM_MODEL=ILD_l5_v02

##
## Generate G4 showers with gun DDSim
##


n=$1
r=$2

export EOS_home=/eos/user/e/eneren
mkdir -p /eos/user/e/eneren/run_$r


echo "-- Running DDSim ${SIM_MODEL} ..."
ddsim \
  --outputFile $EOS_home/run_$r/photon-shower_$n.slcio \
  --compactFile $lcgeo_DIR/ILD/compact/${SIM_MODEL}/${SIM_MODEL}.xml \
  --steeringFile ddsim_steer_gun.py   


Marlin create_root_tree.xml --global.LCIOInputFiles=$EOS_home/run_$r/photon-shower_$n.slcio --MyAIDAProcessor.FileName=$EOS_home/run_$r/pion-shower_$n;

echo $EOS_home/run_$r/photon-shower_$n.root > /mnt/root_path



