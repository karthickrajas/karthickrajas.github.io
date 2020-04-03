---
layout: post
excerpt: Installation notes
images:
  - url: /assets/keras_tensorflow_logo.jpg
---

The procedure given here was tested with GeForce 920MX 2GB Graphic card, Lenovo Ideapad 310.

## **Requirements**

* Python 3.7 version through Anaconda 
* Nvidia Graphics drivers
* Nvidia CUDA toolkit
* Nvidia CuDNN files


## **Download links**


Installing Python by Anaconda will easily set up environments and manage libraries. 

* [Anaconda](https://www.anaconda.com/distribution/) 

  - Select windows and 64 bit system. **Tensorflow supports only 64 bit system.**
  - Complete the installation
  
* [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx?lang=en-us)

  - The site will auto detect the product type, operation system, windows driver etc. If not one can select the relevant options in the drop down list
  - Select the default options and complete the installation, this usually takes some more time for completion
  
* [CUDA Toolkit](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10)

  - Select windows, x64 architecture, version 10, installer type
  - It is recommended to un-select visual studio integration since this might cause an issue if you are running no compatible versions
  - Complete the installation
  
* [CuDNN](https://developer.nvidia.com/cudnn )

  - To download CuDNN, one might need a developer account, which can be registered for free
  - Once registered, you will be redirected to the CuDNN page, choose the relevant version based on the CUDA installed in the previous step
  - After downloading, unzip the CuDNN files and move all the CuDNN files into the cuda toolkit directory. 
  - Keep all the Cuda files unaltered and just paste the CuDNN files in the respective folders ( bin, include, lib\X64 )
  - The location is usually C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\10.0
  
* Adding Environment variables in system manager

  -  Open command prompt -> open "Run"
  - Issue the "control sysdm.cpl" command
  - Select the Advanced tab at the top of the window
  - Click Environment Variables at the bottom of the window
  - Ensure the following values are set, Variable Name: CUDA_PATH, Variable Value: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0

## **Conda Environment**

In ananconda propmt, type

{% highlight ruby %}
conda create -n tensorflow python=3.7.0
{% endhighlight %}

then activate the environment by typing

{% highlight ruby %}
activate tensorflow
{% endhighlight %}

Then one can finally install the gpu version directly by
{% highlight ruby %}
pip install tensorflow-gpu
{% endhighlight %}

## **Testing the GPU**

In anaconda prompt, type

{% highlight ruby %}
activate tensorflow
python
import tensorflow as tf 
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
{% endhighlight %}

We will be able to see a verbose log displaying the details of the gpu.

In the recent version of tensorflow, the following command returns whether the tensorflow can access the GPU.

{% highlight ruby %}
import tensorflow as tf
tf.test.is_gpu_available(
    cuda_only=False,
    min_cuda_compute_capability=None
)
{% endhighlight %}

