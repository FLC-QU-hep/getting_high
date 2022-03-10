#!/usr/bin/env python3
# Copyright 2019 The Kubeflow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from tarfile import RECORDSIZE
import kfp
from kfp import dsl
from kfp.components import InputPath, InputTextFile, InputBinaryFile, OutputPath, OutputTextFile, OutputBinaryFile
from kubernetes import client as k8s_client

eos_host_path = k8s_client.V1HostPathVolumeSource(path='/var/eos')
eos_volume = k8s_client.V1Volume(name='eos', host_path=eos_host_path)
eos_volume_mount = k8s_client.V1VolumeMount(name=eos_volume.name, mount_path='/eos')

krb_secret = k8s_client.V1SecretVolumeSource(secret_name='krb-secret')
krb_secret_volume = k8s_client.V1Volume(name='krb-secret-vol', secret=krb_secret)
krb_secret_volume_mount = k8s_client.V1VolumeMount(name=krb_secret_volume.name, mount_path='/secret/krb-secret-vol')


def sim(pname, rname):
    return dsl.ContainerOp(
                    name='Simulation',
                    image='ilcsoft/ilcsoft-centos7-gcc8.2:v02-01-pre',
                    command=[ '/bin/bash', '-c'],
                    arguments=['cd /home && git clone --branch kf_pipelines https://github.com/FLC-QU-hep/getting_high.git && \
                                git clone --branch v02-01-pre02 https://github.com/iLCSoft/ILDConfig.git && \
                                cp /secret/krb-secret-vol/krb5cc_1000 /tmp/krb5cc_0 && ls && \
                                chmod 600 /tmp/krb5cc_0 && \
                                cd $PWD/ILDConfig/StandardConfig/production && \
                                cp /home/getting_high/kf_pipelines/* . && \
                                chmod +x ./generateG4EOS.sh && ./generateG4EOS.sh "$0" "$1" ', pname, rname],
                    file_outputs={
                        'metadata': '/mnt/root_path'
                    }

    ).add_volume(eos_volume).add_volume_mount(eos_volume_mount).add_volume(krb_secret_volume).add_volume_mount(krb_secret_volume_mount)    

  

def convert_hdf5(rootFile, pname, rname):
    return dsl.ContainerOp(
                    name='hdf5 conversion',
                    image='engineren/pytorch:latest',
                    command=[ '/bin/bash', '-c'],
                    arguments=['git clone --branch kf_pipelines https://github.com/FLC-QU-hep/getting_high.git && cd getting_high/kf_pipelines \
                                && cp /secret/krb-secret-vol/krb5cc_1000 /tmp/krb5cc_0 && chmod 600 /tmp/krb5cc_0 \
                                && python create_hdf5EOS.py --ncpu 4 --rootfile "$0" --outputR "$1" --outputP "$2" --branch photonSIM --batchsize 100', rootFile, rname, pname],
                    file_outputs={
                        'metadata': '/mnt/h5_path'
                    }
    ).add_volume(eos_volume).add_volume_mount(eos_volume_mount).add_volume(krb_secret_volume).add_volume_mount(krb_secret_volume_mount)   


def correction(hdf5File, pname, rname):
    return dsl.ContainerOp(
                    name='hdf5 correction',
                    image='engineren/pytorch:latest',
                    command=[ '/bin/bash', '-c'],
                    arguments=['git clone --branch kf_pipelines https://github.com/FLC-QU-hep/getting_high.git && cd getting_high/kf_pipelines \
                                && cp /secret/krb-secret-vol/krb5cc_1000 /tmp/krb5cc_0 && chmod 600 /tmp/krb5cc_0 \
                                && python correctionsEOS.py --input $0 --outputR "$1" --outputP "$2" --batchsize 10 --minibatch 100', hdf5File, rname, pname]
            
    ).add_volume(eos_volume).add_volume_mount(eos_volume_mount).add_volume(krb_secret_volume).add_volume_mount(krb_secret_volume_mount)   





@dsl.pipeline(
    name='ILDEventGen_Getting_high',
    description='Event Simulation and Reconstruction'
)

def sequential_pipeline():
    """A pipeline with sequential steps."""

   
    ## submit many jobs without control plots
      
    for i in range(1,5):
        runN = 'getting_high_100GeV'
        simulation = sim(str(i), runN)
        inptLCIO = dsl.InputArgumentPath(simulation.outputs['metadata'])
        hf5 = convert_hdf5(inptLCIO, str(i), runN)
        hf5Inpt = dsl.InputArgumentPath(hf5.outputs['metadata'])
        c = correction(hf5Inpt, str(i), runN)
   
    
    
    



   
    
    

if __name__ == '__main__':
    kfp.compiler.Compiler().compile(sequential_pipeline, __file__ + '.yaml')
