This is an abridged and slightly modified version of the implementation of the paper "Towards Faster and Stabilized GAN Training for High-fidelity Few-shot Image Synthesis".  The original implementation by the paper author's can be found here https://github.com/odegeasslbc/FastGAN-pytorch.

Our trained pytorch model can be found at `/candescence/GAN`.

The python scripts here can allow you to interact with our model in various ways:


* eval.py: generate images from our trained generator.  Run `python eval.py --chpt /candescence/GAN/model.pth`

* generate_video.py: generate a continuous video from the interpolation of generated images.  Run `python generate_video.py`

* models.py: contains the pytorch implementation of the GAN.

* operation.py: contains helper functions.


The images used to train the GAN can be found at `/candescence/GAN/images`.
