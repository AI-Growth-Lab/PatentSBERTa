# Patent-Classification
Hybrid Model for Patent Classification using Augmented SBERT and KNN
# Purpose: 
This study aims to provide a hybrid approach for patent claim classification with Sentence-BERT (SBERT) and K Nearest Neighbours (KNN) and explicitly focuses on the patent’s claims. Patent classification is a multi-label classification task in which the number of labels can be greater than 640 at the subclass level. The proposed framework predicts individual input patent class and subclass based on finding top k semantic similarity patents. 
# Design/Methodology/Approach: 
The study uses transformer models based on Augmented SBERT and RoBERTa. We use a different approach to predict patent classification by finding top k similar patent claims and using the KNN algorithm to predict patent class or subclass. Besides, in this study, we just focus on patent claims, and  in the future study, we add other appropriate parts of patent documents. 
# Findings:
The findings suggest the relevance of hybrid models to predict multi-label classification based on text data. In this approach, we used the Transformer model as the distance function in KNN, and proposed a new version of KNN based on Augmented SBERT. 
Practical Implications: The presented framework provides a practical model for patent classification. In this study, we predict the class and subclass of the patent based on semantic claims similarity. The end-user interpretability of the results is one of the essential positive points of the model. 
# Originality/Value:
The main contribution of the study included: 1) Using the Augmented approach for fine-tuning SBERT by in-domain supervised patent claims data. 2) Improving results based on a hybrid model for patent classification. The best result of F1-score at the subclass level was > 69%) Proposing the practical model with high interpretability of results.
# Index Terms
Augmented SBERT, Patent’s claims, RoBERTa, Classification, Hybrid model
# See also
USPTO PatentsView
PatentsView API
PatentsView Query Language
