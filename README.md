## Attention-based Deep Multiple Instance Learning
This is a re-implementation of ICML 2018 paper "Attention-based Deep Multiple Instance Learning" (https://128.84.21.199/pdf/1802.04712.pdf). 

I believe it is a very interesting work and so I built it with Keras using Tensorflow backend. I wrote attention layers described in the paper and did experiments in colon images with 10-fold cross validation. I got the very close average accuracy described in the paper and visualization results can be seen as below. Parts of codes are from https://github.com/yanyongluan/MINNs.

When train the model, we only use the image-level label (0 or 1 to see if it is a cancer image). The attention layer can provide an interpretation of the decision by presenting only a small subset of positive patches.

---

### Results from my implementation

<p align="center">
  <img align="center" src="result.png" width="1000">
</p>

### Dataset
- Colon cancer dataset [[Data]](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/crchistolabelednucleihe/)

I didn't put data here but you can download them from the above link and set up them according to my codes. If you have any problem, please feel free to contact me.
