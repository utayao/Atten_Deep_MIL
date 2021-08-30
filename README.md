## Attention-based Deep Multiple Instance Learning

Attention-based Deep Multiple Instance Learning could be applied in a wide range of medical imaging applications. Supported by the project "[Deep Learning for Survival Prediction](http://ranger.uta.edu/~huang/R_Survival.htm)"@[UTA-SMILE](http://ranger.uta.edu/~huang/), I wrote the **Keras** version of ICML 2018 paper "Attention-based Deep Multiple Instance Learning" (https://arxiv.org/pdf/1802.04712.pdf) in this repo to share the solution for Keras users. 

The official Pytorch implementation can be found [here](https://github.com/AMLab-Amsterdam/AttentionDeepMIL). I built it with **Keras** using Tensorflow backend. I wrote attention layers described in the paper and did experiments in colon images with 10-fold cross validation. I got the very close average accuracy described in the paper and visualization results can be seen as below. Parts of codes are from https://github.com/yanyongluan/MINNs.

When train the model, we only use the image-level label (0 or 1 to see if it is a cancer image). The attention layer can provide an interpretation of the decision by presenting only a small subset of positive patches.

---

### Results from my implementation

<p align="center">
  <img align="center" src="result.png" width="1000">
</p>

### Dataset
- Colon cancer dataset [[Data]](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/crchistolabelednucleihe/)
- Processed patches [[Google Drive]](https://drive.google.com/file/d/1RcNlwg0TwaZoaFO0uMXHFtAo_DCVPE6z/view?usp=sharing)

I put my processed data here and you can also set up according to the paper. If you have any problem, please feel free to contact me.

---
### Applications

Several applications can be found recently. I will summarize them in the following and the first one is our recent work.

 - Jiawen Yao, Xinliang Zhu, et al. ["Whole slide images based cancer survival prediction using attention guided deep multiple instance learning networks"](https://www.sciencedirect.com/science/article/abs/pii/S1361841520301535?dgcid=rss_sd_all), Medical Image Analysis, 101789, 2020.[[PDF]](https://arxiv.org/pdf/2009.11169.pdf) [[Code]](https://github.com/uta-smile/DeepAttnMISL)
<p align="center">
  <img align="center" src="https://camo.githubusercontent.com/1f2a461a631d381a19905e87638440253cd86e44/68747470733a2f2f6172732e656c732d63646e2e636f6d2f636f6e74656e742f696d6167652f312d73322e302d53313336313834313532303330313533352d6678315f6c72672e6a7067" width="600">
</p>

Other important work used multiple-instance learning include
 - Jianan Chen, Helen M. C. Cheung, Laurent Milot and [Anne L. Martel](http://martellab.com/) "[AMINN: Autoencoder-based Multiple Instance Neural Network Improves Outcome Prediction of Multifocal Liver Metastases](https://arxiv.org/pdf/2012.06875.pdf)", MICCAI 2021.  [[Code]](https://github.com/martellab-sri/AMINN)
    - The autoencoder-based multiple instance neural network is deveopled to address multifocality, i.e. to incorporate features from all lesions for prediction/classification.

 - Ole-Johan Skrede et al., “[Deep learning for prediction of colorectal cancer outcome: a discovery and validation study](https://www.thelancet.com/journals/lancet/article/PIIS0140-6736(19)32998-8/fulltext)", the Lancet, 2020.
    - Used [Noisy-AND MIL pooling](https://academic.oup.com/bioinformatics/article/32/12/i52/2288769) to get WSI-level representation.

  - Shujun Wang, Yaxi Zhu, et al. ["RMDL: Recalibrated multi-instance deep learning for whole slide gastric image classification"](https://arxiv.org/abs/2010.06440), Medical Image Analysis, Volume 58, December 2019, 101549. [[Code]](https://github.com/EmmaW8/RMDL)
---
### Contact
If you have any questions about this code, I am happy to answer your issues or emails (to yjiaweneecs@gmail.com).

I plan to review recent work using Deep MIL techniques in medical imaging and Your suggestions are very welcome ! 

### Acknowledgments
--------------------
The work conducted by [Jiawen Yao](https://utayao.github.io/) was funded by Grants from the [UTA-SMILE Lab](https://github.com/uta-smile).

