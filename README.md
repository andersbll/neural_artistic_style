## Neural Artistic Style in Python

Implementation of [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576).


### Requirements
 - [DeepPy](http://github.com/andersbll/deeppy), Deep learning in Python.
 - [CUDArray](http://github.com/andersbll/cudarray), CUDA-accelerated NumPy.
 - [Pretrained VGG 19 model](http://www.vlfeat.org/matconvnet/pretrained), choose *imagenet-vgg-verydeep-19*.


### Example
Execute

    python neural_artistic_style.py --subject images/tuebingen.jpg --style images/starry_night.jpg

### Help
List command line options with

    python neural_artistic_style.py -h
