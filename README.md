# EA-wLTL (Work in Progress)
Experimenting combining Euclidean Alignment (EA) and weighted LTL to classify MI-based EEG 

All results shown are still being developed

## Non-EA and EA EEG trials of one trial
This is a comparison of non-aligned vs aligned EEG trials from a single trial of all electrodes 

![result]("./result.png")

## Visualization of non-alinged features vs aligned features using t-sne
![result]("./nonEA_vs_EA.png")
  
*Not the expected result, the result for EA for target subject (red dot) is expected to more scattered*

## Final results
Comparison of four difference model they are:
1. SVM
2. SVM + EA
3. Weighted Logistic Transfer Learning (wLTL)
4. Weighted Logistic Transfer Learning + EA (wLTL + EA)

![SNE_plot]("SNE_plot.png")

Classification of left and right hand imagery task.
Each model is trained only with 10 trials of each subject (5 samples of each class right and left hand).

## Reference
1. He, H., & Wu, D. (2020). Transfer Learning for Brain-Computer Interfaces: A Euclidean Space Data Alignment Approach. IEEE Transactions on Biomedical Engineering, 67(2), 399–410. https://doi.org/10.1109/TBME.2019.2913914
2. Wu, D., Peng, R., Huang, J., & Zeng, Z. (2020). Transfer Learning for Brain-Computer Interfaces: A Complete Pipeline. 1–9. http://arxiv.org/abs/2007.03746
3. Azab, A. M., Mihaylova, L., Ang, K. K., & Arvaneh, M. (2019). Weighted Transfer Learning for Improving Motor Imagery-Based Brain-Computer Interface. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 27(7), 1352–1359. https://doi.org/10.1109/TNSRE.2019.2923315


