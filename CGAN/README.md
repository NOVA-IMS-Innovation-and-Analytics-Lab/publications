# Effective data generation for imbalanced learning using Conditional Generative Adversarial Networks

### Highlights
* Application of conditional Generative Adversarial Networks as oversampling algorithm.
* Generates minority class samples by recovering the training data distribution.
* Outperforms various standard oversampling algorithms.
* Performance advantage of the proposed method increases remains stable with higher imbalance ratios.

### Abstract
Learning from imbalanced datasets is a frequent but challenging task for standard classification algorithms. Although there are different strategies to address this problem, methods that generate artificial data for the minority class constitute a more general approach compared to algorithmic modifications. Standard oversampling methods are variations of the SMOTE algorithm, which generates synthetic samples along the line segment that joins minority class samples. Therefore, these approaches are based on local information, rather on the overall minority class distribution. Contrary to these algorithms, in this paper the conditional version of Generative Adversarial Networks (cGAN) is used to approximate the true data distribution and generate data for the minority class of various imbalanced datasets. The performance of cGAN is compared against multiple standard oversampling algorithms. We present empirical results that show a significant improvement in the quality of the generated data when cGAN is used as an oversampling algorithm.

### Link to GAN and cGAN
[TensorFlow implementation](https://github.com/gdouzas/generative-adversarial-nets)

### Link to the implemented oversampling algorithm
[Oversampling algorithm](https://github.com/gdouzas/imbalanced-tools/blob/master/imbtools/algorithms/cgan_oversampler.py)