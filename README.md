## Neural Artistic Style in Python

Implementation of [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576). A method to transfer the style of one image to the subject of another image.


### Requirements
 - [DeepPy](http://github.com/andersbll/deeppy), Deep learning in Python.
 - [CUDArray](http://github.com/andersbll/cudarray) with [cuDNN](https://developer.nvidia.com/cudnn), CUDA-accelerated NumPy.
 - [Pretrained VGG 19 model](http://www.vlfeat.org/matconvnet/pretrained), choose *imagenet-vgg-verydeep-19*.


### Examples
Execute

    python neural_artistic_style.py --subject images/tuebingen.jpg --style images/starry_night.jpg

The two inputs are

<p align="center">
Subject:
<img src="https://github.com/andersbll/neural_artistic_style/blob/master/images/tuebingen.jpg?raw=true" width="30%"/>
Style:
<img src="https://github.com/andersbll/neural_artistic_style/blob/master/images/starry_night.jpg?raw=true" width="30%"/>
</p>

The output becomes:
<p align="center">
<img src="https://github.com/andersbll/neural_artistic_style/blob/master/images/tuebingen-starry_night.jpg?raw=true" width="30%"/>
</p>

We can also choose a (younger version) of HM the Queen of Denmark as subject and paint her using different styles. Click the images to see the full size.

**Subject**
<p align="center">
<img src="https://github.com/andersbll/neural_artistic_style/blob/master/images/margrethe.jpg?raw=true" width="20%"/>
</p>

**Styles**
<p align="center">
<img src="https://github.com/andersbll/neural_artistic_style/blob/master/images/lundstroem.jpg?raw=true" width="18%"/>
<img src="https://github.com/andersbll/neural_artistic_style/blob/master/images/donelli.jpg?raw=true" width="18%"/>
<img src="https://github.com/andersbll/neural_artistic_style/blob/master/images/picasso.jpg?raw=true" width="18%"/>
<img src="https://github.com/andersbll/neural_artistic_style/blob/master/images/groening.jpg?raw=true" width="18%"/>
<img src="https://github.com/andersbll/neural_artistic_style/blob/master/images/skrik.jpg?raw=true" width="18%"/>
</p>

**Outputs**
<p align="center">
<img src="https://github.com/andersbll/neural_artistic_style/blob/master/images/margrethe_lundstroem.jpg?raw=true" width="18%"/>
<img src="https://github.com/andersbll/neural_artistic_style/blob/master/images/margrethe_donelli.jpg?raw=true" width="18%"/>
<img src="https://github.com/andersbll/neural_artistic_style/blob/master/images/margrethe_picasso.jpg?raw=true" width="18%"/>
<img src="https://github.com/andersbll/neural_artistic_style/blob/master/images/margrethe_groening.jpg?raw=true" width="18%"/>
<img src="https://github.com/andersbll/neural_artistic_style/blob/master/images/margrethe_skrik.jpg?raw=true" width="18%"/>
</p>


### Help
List command line options with

    python neural_artistic_style.py --help
