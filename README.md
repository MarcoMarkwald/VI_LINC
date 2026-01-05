# VI_LINC
---
We provide our code to support the reproducibility of results
presented in the VI_Linc paper. However, we do not share models
or datasets. The sources of the datasets we used are referenced 
in the paper.

## 📌 Abstract / Overview
Imbalanced graph node classification is a highly relevant problem in various
real-world applications, including fraud detection, spam filtering, and network
intrusion detection. In these settings, detecting minority class nodes is parti-
cularly challenging due to the inherent structural diversity of the patterns and
the limited availability of labeled training data. Conventional graph learning
methods are not explicitly designed for imbalanced settings and often fail to
generalize class-specific patterns from sparse and heterogeneous examples. To
address these challenges, we propose VI-LINC, a novel approach for visual repre-
sentation learning on graphs tailored to imbalanced node classification. VI-LINC
introduces a structured visual representation of local graph neighborhoods as
a basis for forming class-specific pattern clusters in a visual embedding space,
improving class-discriminative pattern extraction and generalization. Extensive
experiments on three datasets demonstrate that VI-LINC consistently outper-
forms state-of-the-art methods for imbalanced node classification, achieving an
increase of up to 15.45% in precision, 2.13% in recall, and 43.34% f1-score for the
macro scores on real-world data. For graph patterns with more complex struc-
ture - specifically nodes with 3 or more 1-hop neighbours, VI-LINC achieves an
18.17% improvement in precision, 8.57% in recall, and 15.09% in f1-score on the
minority class for a real-world dataset.


## 🗂️ Directory Structure
├── Vi_Linc_main.py
├── requirements.txt 
└── README.md
