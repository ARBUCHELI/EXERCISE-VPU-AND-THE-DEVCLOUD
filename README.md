# EXERCISE-VPU-AND-THE-DEVCLOUD

## This Content was Created by Intel Edge AI for IoT Developers UDACITY Nanodegree. (Solution of the exercise and adaptation as a repository: Andrés R. Bucheli.)

# Exercise: VPU and the DevCloud

Requesting a CPU with a VPU (Intel® NCS2) on Intel's DevCloud, loading a model on the Intel® NCS2, and running inference on an image using both a CPU and IGPU.

In this exercise, you will do the following:

1. Write a Python script to load a model and run inference 100 times on a device on Intel's DevCloud.
    * Calculate the time it takes to load the model.
    * Calculate the time it takes to run inference 100 times.
2. Write a shell script to submit a job to Intel's DevCloud.
2. Submit a job using <code>qsub</code> on an <strong>IEI Tank-870</strong> edge node, run <code>liveQStat</code> to view the status of your submitted jobs, then retrieve and view
the results from your job.
  * One job using CPU as the device.
  * One job using GPU as the device.
  * One job using VPU as the device.
4. Plot and compare the results using bar graphs with matplotlib for the following metrics:
  * Model Loading Time
  * Inference Time
  * Frames Per Second (FPS)
  
<strong>IMPORTANT: Set up paths so we can run Dev Cloud utilities</strong>
You must run this every time you enter a Workspace session.

<pre><code>
%env PATH=/opt/conda/bin:/opt/spark-2.4.3-bin-hadoop2.7/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/intel_devcloud_support
import os
import sys
sys.path.insert(0, os.path.abspath('/opt/intel_devcloud_support'))
sys.path.insert(0, os.path.abspath('/opt/intel')
</code></pre>

# The Model
We will be using the <code>vehicle-license-plate-detection-barrier-0106</code> model for this exercise.

Remember to use the appropriate model precisions for each device:

  * CPU -<code>FP32</code>
  * IGPU - <code>FP16</code>
  * VPU - <code>FP16</code>
  
The model has already been downloaded for you in the <code>/data/models/intel</code> directory on Intel's DevCloud.

We will be running inference on an image of a car. The path to the image is <code>/data/resources/car.png</code>.

# Step 1: Creating a Python Script
The first step is to create a Python script that you can use to load the model and perform inference. We'll use the <code<%%writefile</code> magic to create a Python file 
called <code>inference_on_device.py</code>. In the next cell, you will need to complete the <code>TODO</code> items for this Python script.

<code>TODO</code> items:

1. Load the model

2. Get the name of the input node

3. Prepare the model for inference (create an input dictionary)

4. Run inference 100 times in a loop

<pre><code>
%%writefile inference_on_device.py

import time
import numpy as np
import cv2
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IECore
import argparse

def main(args):
    model=args.model_path
    model_weights=model+'.bin'
    model_structure=model+'.xml'
    
    start=time.time()
    
    # TODO: Load the model
    model=IENetwork(model_structure, model_weights)

    core = IECore()
    net = core.load_network(network=model, device_name=args.device, num_requests=1)
    
    load_time=time.time()-start
    print(f"Time taken to load model = {load_time} seconds")
    
    # Get the name of the input node
    input_name=next(iter(model.inputs))
    # Reading and Preprocessing Image
    input_img=cv2.imread('/data/resources/car.png')
    input_img=cv2.resize(input_img, (300,300), interpolation = cv2.INTER_AREA)
    input_img=np.moveaxis(input_img, -1, 0)

    # TODO: Prepare the model for inference (create input dict etc.)
    input_dict={input_name:input_img}
    start=time.time()
    for _ in range(100):
        # TODO: Run Inference in a Loop
        net.infer(input_dict)
    inference_time=time.time()-start
    fps=100/inference_time
    print(f"Time Taken to run 100 inference is = {inference_time} seconds")
    
    with open(f"/output/{args.path}.txt", "w") as f:
        f.write(str(load_time)+'\n')
        f.write(str(inference_time)+'\n')
        f.write(str(fps)+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--device', default=None)
    parser.add_argument('--path', default=None)
    
    args=parser.parse_args() 
    main(args)
</code></pre>

# Step 2: Creating a Job Submission Script
To submit a job to the DevCloud, you'll need to create a shell script. Similar to the Python script above, we'll use the <code>%%writefile</code> magic command to create a shell
script called <code>inference_model_job.sh</code>. In the next cell, you will need to complete the <code>TODO</code> items for this shell script.

<code>TODO</code> items:

1. Create three variables:
    * <code>DEVICE</code> - Assign the value as the first argument passed into the shell script.
    * <code>MODELPATH</code> - Assign the value as the second argument passed into the shell script.
    * <code>SAVEPATH</code> - Assign the value as the third argument passed into the shell script.
2. Call the Python script using the three variable values as the command line argument

<pre><code>
%%writefile inference_model_job.sh
#!/bin/bash

exec 1>/output/stdout.log 2>/output/stderr.log

mkdir -p /output

# TODO: Create DEVICE variable
# TODO: Create MODELPATH variable
# TODO: Create SAVEPATH variable
DEVICE=$1
MODELPATH=$2
SAVEPATH=$3

# TODO: Call the Python script
python3 inference_on_device.py  --model_path ${MODELPATH} --device ${DEVICE} --path ${SAVEPATH}

cd /output

tar zcvf output.tgz * # compresses all files in the current directory (output)
</code></pre>

# Step 3: Submitting a Job to Intel's DevCloud
In the next three sub-steps, you will write your <code>!qsub</code> commands to submit your jobs to Intel's DevCloud to load your model and run inference on the <strong>IEI 
Tank-870</strong> edge node with an <code>Intel Core i5</code> CPU and an <code>Intel Neural Compute Stick 2</code> VPU.

Your <code>!qsub</code> command should take the following flags and arguments:

1. The first argument should be the shell script filename
2. <code>-d</code> flag - This argument should be <code>.</code>
3. <code>-l</code> flag - This argument should request an edge node with an <strong>IEI Tank-870</strong>. The default quantity is 1, so the 1 after <code>nodes</code> is
optional.

  * <strong>Intel Core i5 6500TE</strong> for your <code>CPU</code>.
  * <strong>Intel HD Graphics 530</strong> for your <code>GPU</code>.
  * <strong>Intel Neural Compute Stick 2</strong> for your <code>VPU</code>.
  
4. <code>-F</code> flag - This argument should contain the three values to assign to the variables of the shell script:
    * <strong>DEVICE</strong> - Device type for the job: <code>CPU,GPU or MYRIAD</code>.
    * <strong>MODELPATH</strong> - Full path to the model for the job. As a reminder, the model is located in <code>/data/models/intel</code>.
    * <strong>SAVEPATH</strong> - Name of the file you want to save the performance metrics as. These should be named as the following:
        * <code>cpu_stats</code> for the <code>CPU</code> job
        * <code>vpu_stats</code> for the <code>VPU</code> job
        * <code>gpu_stats</code> for the <code>GPU</code> job
        
<strong>Note</strong>: There is an optional flag, <code>-N</code>, you may see in a few exercises. This is an argument that only works on Intel's DevCloud that allows you to name your job
submission. This argument doesn't work in Udacity's workspace integration with Intel's DevCloud.

# Step 3a: Running on the NCS2
In the cell below, write the qsub command that will submit your job to the VPU (NCS2).

<pre><code>
vpu_job_id_core = !qsub inference_model_job.sh -d . -l nodes=tank-870:i5-6500te:intel-ncs2 -F "MYRIAD /data/models/intel/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106 vpu_stats" -N store_core 
print(vpu_job_id_core[0])
</code></pre>

# Check Job Status
To check on the job that was submitted, use <code>liveQStat</code> to check the status of the job. The cell is locked until this finishes polling 10 times or you can interrupt the kernel to
stop it by pressing the stop button at the top:

Column <code>S</code> shows the state of your running jobs.

For example:

  * If <code>JOB ID</code> is in Q state, it is in the queue waiting for available resources.
  * If <code>JOB ID</code> is in R state, it is running.

<pre><code>
import liveQStat
liveQStat.liveQStat()
</code></pre>

<strong>Get Results</strong>
Run the next cell to retrieve your job's results.

<pre><code>
import get_results

get_results.getResults(vpu_job_id_core[0], filename="output.tgz", blocking=True)
</code></pre>

<strong>Unpack your output files and view stdout.log</strong>

<code>!tar zxf output.tgz</code>
<code>!cat stdout.log</code>

<strong>View stderr.log</strong>
This can be used for debugging

<code>!cat stderr.log</code>

# Step 3b: Running on the CPU
In the cell below, write the qsub command that will submit your job to the CPU.

<pre><code>
cpu_job_id_core = !qsub inference_model_job.sh -d . -l nodes=tank-870:i5-6500te -F "CPU /data/models/intel/vehicle-license-plate-detection-barrier-0106/FP32/vehicle-license-plate-detection-barrier-0106 cpu_stats" -N store_core 
print(cpu_job_id_core[0])
</code></pre>

# Check Job Status
To check on the job that was submitted, use <code>liveQStat</code> to check the status of the job. The cell is locked until this finishes polling 10 times or you can interrupt
the kernel to stop it by pressing the stop button at the top:

Column <code>S</code> shows the state of your running jobs.

For example:

  * If <code>JOB ID</code> is in Q state, it is in the queue waiting for available resources.
  * If <code>JOB ID</code> is in R state, it is running.
  
<pre><code>
import liveQStat
liveQStat.liveQStat()
</code></pre>

<strong>Get Results</strong>
Run the next cell to retrieve your job's results.

<pre><code>
import get_results

get_results.getResults(cpu_job_id_core[0], filename="output.tgz", blocking=True)
</code></pre>

<strong>Unpack your output files and view stdout.log</strong>

<code>!tar zxf output.</code>
<code>!cat stdout.log</code>

<strong>View stderr.log</strong>
This can be used for debugging

# Step 3c: Running on the GPU
In the cell below, write the qsub command that will submit your job to the GPU.

<pre><code>
gpu_job_id_core = !qsub inference_model_job.sh -d . -l nodes=tank-870:i5-6500te:intel-hd-530 -F "GPU /data/models/intel/vehicle-license-plate-detection-barrier-0106/FP16/vehicle-license-plate-detection-barrier-0106 gpu_stats" -N store_core 
print(gpu_job_id_core[0])
</code></pre>

<strong>Check Job Status</strong>
To check on the job that was submitted, use <code>liveQStat</code> to check the status of the job. The cell is locked until this finishes polling 10 times or you can interrupt 
the kernel to stop it by pressing the stop button at the top:

Column <code>S</code> shows the state of your running jobs.

For example:

  * If <code>JOB ID</code> is in Q state, it is in the queue waiting for available resources.
  * If <code>JOB ID</code> is in R state, it is running.

<pre><code>
import liveQStat
liveQStat.liveQStat()
</code></pre>

<strong>Results</strong>
Run the next cell to retrieve your job's results.

<pre><code>
import get_results

get_results.getResults(gpu_job_id_core[0], filename="output.tgz", blocking=True)
</code></pre>

<strong>Unpack your output files and view stdout.log</strong>

<code>!tar zxf output.tgz</code>
<code>!cat stdout.log</code>

<strong>View stderr.lo</strong>
This can be used for debugging

<code>!cat stderr.log</code>

# Step 4: Plot and Compare Results
Run the cell below to plot and compare the results.

<code>import matplotlib.pyplot as plt</code>
<pre><code>
def plot(labels, data, title, label):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.bar(labels, data)
    
def read_files(paths, labels):
    load_time=[]
    inference_time=[]
    fps=[]
    
    for path in paths:
        if os.path.isfile(path):
            f=open(path, 'r')
            load_time.append(float(f.readline()))
            inference_time.append(float(f.readline()))
            fps.append(float(f.readline()))

    plot(labels, load_time, 'Model Load Time', 'seconds')
    plot(labels, inference_time, 'Inference Time', 'seconds')
    plot(labels, fps, 'Frames per Second', 'Frames')

paths=['vpu_stats.txt', 'gpu_stats.txt', 'cpu_stats.txt']
read_files(paths, ['VPU', 'GPU', 'CPU'])
</code></pre>

![image](https://raw.githubusercontent.com/ARBUCHELI/EXERCISE-VPU-AND-THE-DEVCLOUD/master/downloada.png)

![image](https://raw.githubusercontent.com/ARBUCHELI/EXERCISE-VPU-AND-THE-DEVCLOUD/master/downloadb.png)

![image](https://raw.githubusercontent.com/ARBUCHELI/EXERCISE-VPU-AND-THE-DEVCLOUD/master/downloadc.png)

## Solution of the exercise and adaptation as a Repository: Andrés R. Bucheli.















