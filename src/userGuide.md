# Bat SSL
<img align="right" width="128" height="128" src="img/bat.jpg">  


## What is Bat SSL?

Bat SSL is a Python-based tool designed for the classification of bat echolocation calls within full-spectrum audio recordings. The code embodies the machine learning pipeline I presented in my report, [Monitoring Bat Activity in Audio with Self-Supervised Learning](https://github.com/Andydiantu/BatSSL/blob/main/Final_Report.pdf).

## The Intersection of Bats and Self-Supervised Learning

Bats serve as exceptional bioindicators. The dynamics of their population and species offer a lens to assess anthropogenic impacts on biodiversity and the environment.

However, manually labeling bat echolocation calls is arduous. The self-supervised approach leverages vast amounts of unlabeled data, enhancing the performance and mitigating this challenge.

## Performance Insights

The table below demonstrates that with self-supervised pretraining, Bat SSL achieves competitive accuracy using just half of the labeled data when juxtaposed with its fully supervised counterpart. Remarkably, with a mere 10% of labeled data, the accuracy only dips by 9%.

|              | Supervised | BatSSL with 50% Labeled Data | BatSSL with 10% Labeled Data |
|--------------|------------|-----------------------------|-----------------------------|
| Accuracy (%) |    75.92   |           75.49             |           66.19             |

In its self-supervised pretraining phase, Bat SSL effectively captures visual representations on par with those gleaned by the supervised baseline, all without leveraging any labeled data at the genus level. This prowess is evident in the T-SNE plots provided below.

<img  src="img/sne.jpg">  
