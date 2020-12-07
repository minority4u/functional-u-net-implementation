# Functional implementation of the U-Net

Ronnebergers U-Net architecture with small adjustments. All parameters are injectable. This implementation is based on tensorflow 2.x and reached state of the art Dice scores.

The segmentation network has a U-Net architecture with four down-sampling and four respective up-samplingblocks (based on the original U-Net from Ronneberger et al) with some small adjustments:  batchnormalizationlayer after each non-linear activation, higher dropout rates, ELU activation instead of RELU, Dice loss function,data augmentation and no resampling to force the network to learn different physical representations to increasethe generalisation of the network.

# This implementation was used for the following publication:

## How well do U-Net-based segmentation trained on adult cardiac magnetic resonance imaging data generalise to rare congenital heart diseases for surgical planning?

[arxiv link](https://arxiv.org/abs/2002.04392)


---

Please cite this paper if you use this code:


> @misc{koehler2020unetbased,
    title={How well do U-Net-based segmentation trained on adult cardiac magnetic resonance imaging data generalise to rare congenital heart diseases for surgical planning?},
    author={Sven Koehler and Animesh Tandon and Tarique Hussain and Heiner Latus and Thomas Pickardt and Samir Sarikouch and Philipp Beerbaum and Gerald Greil and Sandy Engelhardt and Ivo Wolf},
    year={2020},
    eprint={2002.04392},
    archivePrefix={arXiv},
    primaryClass={eess.IV}
}
