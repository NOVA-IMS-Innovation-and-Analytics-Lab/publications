## Small Data Oversampling, Improving small data prediction accuracy using the Geometric SMOTE for Small Data algorithm

### Abstract

In the age of the data deluge there are still many domains and applications
restricted to the use of small datasets. The ability to harness these small
datasets to solve problems through the use of supervised learning methods can
have a significant impact in many important areas. The insufficient size of
training data usually results in unsatisfactory performance of machine learning
algorithms. The current research work aims to contribute to mitigate the small
data problem through the creation of artificial instances, which are added to
the training process. The proposed algorithm Geometric SMOTE for Small Data
utilizes the Geometric SMOTE data generation mechanism to generate new high
quality instances. Experimental results present a significant improvement in
accuracy when compared with the use of the initial small dataset and also over
other oversampling techniques such as Random OverSampling, SMOTE and
Borderline SMOTE. These findings show that oversampling research, developed
in the context of imbalanced learning, can also be a valid option to deal
with the small data problem.

## [Imbalanced Learning in Land Cover Classification, improving minority classes' prediction accuracy using the Geometric SMOTE algorithm](https://www.mdpi.com/2072-4292/11/24/3040)

### Abstract

In spite of its importance in sustainable resource management, the automatic
production of Land Use/Land Cover maps continues to be a challenging problem.
The ability to build robust automatic classifiers able to produce accurate maps
can have a significant impact in the way we manage and optimize natural
resources. The difficulty in achieving these results comes from many different
factors, such as data quality and uncertainty, among others. In this paper, we
address the imbalanced learning problem, a common and difficult problem in
remote sensing that affects the quality of classifiers. Having very asymmetric
distributions of the different classes constitutes a significant hurdle for any
classifier. In this work, we propose Geometric-SMOTE as a means of addressing
the imbalanced learning problem in remote sensing. Geometric-SMOTE is a
sophisticated oversampling algorithm which increases the quality of the
generated instances over previous methods, such as Synthetic Minority
Oversampling TEchnique. The performance of Geometric-SMOTE, in the the LUCAS
dataset, is compared to other oversamplers using a variety of classifiers. The
results show that Geometric-SMOTE significantly outperforms all the other
oversamplers and improves the robustness of the classifiers. These results
indicate that, when using imbalanced datasets, remote sensing researchers should
consider the use of these new generation oversamplers to increase the quality of
the classification results.

## [Improving Imbalanced Land Cover Classification with K-Means SMOTE: Detecting and Oversampling Distinctive Minority Spectral Signatures](https://www.mdpi.com/2078-2489/12/7/266)

### Abstract

In spite of its importance in sustainable resource management, the automatic
production of Land Use/Land Cover maps continues to be a challenging problem.
The ability to build robust automatic classifiers able to produce accurate maps
can have a significant impact in the way we manage and optimize natural
resources. The difficulty in achieving these results comes from many different
factors, such as data quality and uncertainty, among others. In this paper, we
address the imbalanced learning problem, a common and difficult problem in
remote sensing that affects the quality of classifiers. Having very asymmetric
distributions of the different classes constitutes a significant hurdle for any
classifier. In this work, we propose Geometric-SMOTE as a means of addressing
the imbalanced learning problem in remote sensing. Geometric-SMOTE is a
sophisticated oversampling algorithm which increases the quality of the
generated instances over previous methods, such as Synthetic Minority
Oversampling TEchnique. The performance of Geometric-SMOTE, in the the LUCAS
dataset, is compared to other oversamplers using a variety of classifiers. The
results show that Geometric-SMOTE significantly outperforms all the other
oversamplers and improves the robustness of the classifiers. These results
indicate that, when using imbalanced datasets, remote sensing researchers should
consider the use of these new generation oversamplers to increase the quality of
the classification results.

## [Improving imbalanced learning through a heuristic oversampling method based on k-means and SMOTE](https://www.sciencedirect.com/science/article/pii/S0020025518304997)

### Abstract

Learning from class-imbalanced data continues to be a common and challenging
problem in supervised learning as standard classification algorithms are
designed to handle balanced class distributions. While different strategies
exist to tackle this problem, methods which generate artificial data to achieve
a balanced class distribution are more versatile than modifications to the
classification algorithm. Such techniques, called oversamplers, modify the
training data, allowing any classifier to the proposed method improves
classification results. Moreover, k-means SMOTE consistently outperforms other
popular oversampling methods. An implementation is made available in the Python
programming language.

## [Self-Organizing Map Oversampling (SOMO) for imbalanced data set learning](https://www.sciencedirect.com/science/article/pii/S0957417417302324)

### Abstract

Learning from imbalanced datasets is challenging for standard algorithms, as
they are designed to work with balanced class distributions. Although there are
different strategies to tackle this problem, methods that address the problem
through the generation of artificial data constitute a more general approach
compared to algorithmic modifications. Specifically, they generate artificial
data that can be used by any algorithm, not constraining the options of the
user. In this paper, we present a new oversampling method, Self-Organizing
Map-based Oversampling (SOMO), which through the application of a
Self-Organizing Map produces a two dimensional representation of the input
space, allowing for an effective generation of artificial data points. SOMO
comprises three major stages: Initially a Self-Organizing Map produces a
two-dimensional representation of the original, usually high-dimensional, space.
Next it generates within-cluster synthetic samples and finally it generates
between cluster synthetic samples. Additionally we present empirical results
that show the improvement in the performance of algorithms, when artificial data
generated by SOMO are used, and also show that our method outperforms various
oversampling methods.

## G-SOMO, an oversampling approach based on Self-Organized Maps and Geometric SMOTE

### Abstract

Traditional supervised machine learning classifiers are challenged to learn
highly skewed data distributions as they are designed to expect classes to
equally contribute to the minimization of the classifiers cost function.
Moreover, the classifiers design expects equal misclassification costs, causing
a bias for underrepresented classes. Thus, different strategies to handle the
issue are proposed by researchers. The modification of the data set has become
a common practice since the procedure is generalizable to all classifiers.
Various algorithms to rebalance the data distribution through the creation of
synthetic instances were proposed in the past.  In this paper, we propose a new
oversampling algorithm named G-SOMO, a method that is adopted from our previous
research. The algorithm identifies optimal areas to create artificial data
instances in an informed manner and utilizes a geometric region during the data
generation to increase variability and to avoid correlation. Our empirical
results on 69 datasets, validated with different classifiers and metrics
against a benchmark of commonly used oversampling methods show that G-SOMO
consistently outperforms competing oversampling methods. The statistical
significance of our results is presented.

## [Geometric SMOTE: A geometrically enhanced drop-in replacement for SMOTE](https://www.sciencedirect.com/science/article/pii/S0020025519305353?via%3Dihub)

### Abstract

Classification of imbalanced datasets is a challenging task for standard
algorithms. Although many methods exist to address this problem in different
ways, generating artificial data for the minority class is a more general
approach compared to algorithmic modifications. SMOTE algorithm, as well as any
other oversampling method based on the SMOTE mechanism, generates synthetic
samples along line segments that join minority class instances. In this paper we
propose Geometric SMOTE (G-SMOTE) as a enhancement of the SMOTE data generation
mechanism. G-SMOTE generates synthetic samples in a geometric region of the
input space, around each selected minority instance. While in the basic
configuration this region is a hyper-sphere, G-SMOTE allows its deformation to a
hyper-spheroid. The performance of G-SMOTE is compared against SMOTE as well as
baseline methods. We present empirical results that show a significant
improvement in the quality of the generated data when G-SMOTE is used as an
oversampling algorithm. An implementation of G-SMOTE is made available in the
Python programming language.

## Comparing the performance of oversampling techniques in combination with a clustering algorithm for imbalanced learning

### Abstract

Imbalanced learning constitutes a recurring problem in machine learning.  
Frequently, practitioners and researchers have access to large amounts of 
data but its imbalanced nature hampers the possibility of building accurate 
robust predictive models. Many problems are characterized by the rare nature 
of the cases of interest, in domains such as health, security, quality control 
business applications, it is frequent to have huge populations of negative 
cases and just a very limited number of positive ones. The prevalence of the 
imbalanced learning problem explains why it continues to be a very active 
research topic, which in fact has grown in recent years. It is known that 
imbalance can exist both between-classes (the imbalance occurring between 
the two classes) and also within-classes (the imbalance occurring between 
sub-clusters of each class). In this later case traditional oversamplers 
will tend to perform poorly. In this paper we focus on testing the relevancy 
of using clustering procedures in combination with oversamplers to improve 
the quality of the results. We perform a series of extensive tests using 
the most popular oversamplers, with and without a preceding clustering 
procedure. The tests confirm that the use of a clustering procedure prior 
to the application of an oversampler will improve the results.

## [Effective data generation for imbalanced learning using conditional generative adversarial networks](https://www.sciencedirect.com/science/article/pii/S0957417417306346)

### Abstract

Learning from imbalanced datasets is a frequent but challenging task for
standard classification algorithms. Although there are different strategies to
address this problem, methods that generate artificial data for the minority
class constitute a more general approach compared to algorithmic modifications.
Standard oversampling methods are variations of the SMOTE algorithm, which
generates synthetic samples along the line segment that joins minority class
samples. Therefore, these approaches are based on local information, rather on
the overall minority class distribution. Contrary to these algorithms, in this
paper the conditional version of Generative Adversarial Networks (cGAN) is used
to approximate the true data distribution and generate data for the minority
class of various imbalanced datasets. The performance of cGAN is compared
against multiple standard oversampling algorithms. We present empirical results
that show a significant improvement in the quality of the generated data when
cGAN is used as an oversampling algorithm.