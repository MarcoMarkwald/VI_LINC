# VI_LINC
---
We provide our code to support the reproducibility of results
presented in the VI_Linc paper. However, we do not share models
or datasets. The sources of the datasets we used are referenced 
in the paper.

## 📌 Abstract / Overview
Imbalanced graph node classification is a highly relevant
problem in various real-world applications, including fraud
detection, spam filtering, and network intrusion detection. In
these settings, detecting minority class nodes is particularly
challenging due to the inherent structural diversity of the pat-
terns and the limited availability of labeled training data. Con-
ventional graph learning methods are not explicitly designed
for imbalanced settings and often fail to generalize class-
specific patterns from sparse and heterogeneous examples. To
address these challenges, we propose VI-LINC, a novel ap-
proach for visual representation learning on graphs tailored to
imbalanced node classification. VI-LINC introduces a struc-
tured visual representation of local graph neighborhoods as
a basis for forming class-specific pattern clusters in a visual
embedding space, improving class-discriminative pattern ex-
traction and generalization. Extensive experiments on real-
world datasets demonstrate that VI-LINC consistently out-
performs state-of-the-art methods for imbalanced node clas-
sification, achieving an increase of up to 3.0% in precision,
7.5% in recall and 5.1% F1-score for the minority class.


## 🗂️ Directory Structure
├── Vi_Linc_main.py
├── requirements.txt 
└── README.md
