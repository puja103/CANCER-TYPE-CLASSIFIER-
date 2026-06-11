# CANCER-TYPE-CLASSIFIER.
INTRODUCTION:

This is a Randon forest model that classifies 5 types of cancers via RNA sequencing using the Cancer Genome Atlas dataset.(801 samples, 20,531 genes).

The 5 cancer classes are BRCA (breast), KIRC (kidney), COAD (colon), LUAD (lung adenocarcinoma), and PRAD (prostate).

STEPS:

1. Download the TCGA PanCan HiSeq dataset (801 samples, 20,531 genes, 5 cancer classes) directly via ucimlrepo.
2. After data preprocessing, it was narrowed to 50 features using VarianceThreshold and the ANOVA-F test
3. An random forest model was trained using the reduced feature dataset.

RESULTS:

1.5-fold CV Accuracy - 99.69% +/- 0.38%

2.Test Accuracy	        98.76%

3.Weighted F1	          98.74%



