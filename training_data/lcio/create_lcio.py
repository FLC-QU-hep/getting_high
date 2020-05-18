import math
import random
from array import array
import json

# --- LCIO dependencies ---
from pyLCIO import UTIL, EVENT, IMPL, IO, IOIMPL

#---- number of events  -----
nevt = 1

outfile = "wgan.slcio"

#--------------------------------------------

wrt = IOIMPL.LCFactory.getInstance().createLCWriter( )

wrt.open( outfile , EVENT.LCIO.WRITE_NEW ) 

print " opened outfile: " , outfile

random.seed()

with open('photon_1evt.json') as f:
    data = json.load(f)


pdg = 22

# write a RunHeader
run = IMPL.LCRunHeaderImpl() 
run.setRunNumber( 0 ) 
run.parameters().setValue("Generator","WGAN")
run.parameters().setValue("PDG", pdg )
wrt.writeRunHeader( run ) 


for j in range( 0, nevt ):

        col = IMPL.LCCollectionVec( EVENT.LCIO.SIMCALORIMETERHIT ) 
        flag =  IMPL.LCFlagImpl(0) 
        flag.setBit( EVENT.LCIO.CHBIT_LONG ) 
        col.setFlag( flag.getFlag() )

        evt = IMPL.LCEventImpl() 
        evt.setEventNumber( j ) 
        evt.addCollection( col , "SimCalorimeterHit" )

        
        sch = IMPL.SimCalorimeterHitImpl()
        ### get from json file, loop over 27k cells
        for i in range(0,27000):
            energy = data[i]['e']
            x = data[i]['x']
            y = data[i]['y']
            z = data[i]['z']
            position = array('f',[x,y,z])
            #print x,y,z,energy
            sch.setPosition(position)
            sch.setEnergy(energy)
            


        ### create SimCalorimeterHit
        col.addElement( sch )
        wrt.writeEvent( evt ) 


wrt.close() 
