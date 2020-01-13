=======================================================================================================================
Comparing the performance of oversampling techniques in combination with a clustering algorithm for imbalanced learning
=======================================================================================================================

Abstract
========

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