# EA-wLTL (Work in Progress)
Experimenting combining Euclidean Alignment (EA) and weighted LTL to classify MI-based EEG 

All results shown are still being developed

## Non-EA and EA EEG trials of one trial
Updated on 25th March, 2021:
- Preprocessed data
- Comparing 6 approaches

This is a comparison of non-aligned (black) vs aligned (red) EEG trials from a single trial of all electrodes  
  
![](/img/nonEA_vs_EA.png)

## Visualization of non-alinged features vs aligned features using t-sne
![SNE_plot](/img/SNE_plot.png)
*Not the expected result, the result for EA for target subject (red dot) is expected to more scattered*

## Effect of EA on LDA and SVM
This section compares effects of doing EA (Euclidean Alignment) using LDA and SVM as classifier, each subject alternately acts as target while the other 8 act as source when EA is applied.

### Objective
1. Despite the model, and whether or not EA is applied, using same number of trials won't improve the result
2. Using non-EA source trials to train target will worsen accuracy
3. using EA source trials to train target will improve accuracy


### Evaluation scheme

![evaluation_scheme](/img/evaluation_scheme_4pat.png)

### Result
![evaluation_scheme](/img/svm_lda_4pat_bar.png)

Conclusion:
1. Objective 1 is proofed by comparing pattern 1 and 2, the difference between the two is negligible, on either classifier.
2. Objective 2 and 3 can be observe by comparing pattern 3 and pattern 4



## Comparing 6 Approaches
Comparison of six different approaches they are:
1. CSP-SVM
2. CSP-LDA
3. EA-CSP-LDA
4. EA-CSP-SVM
5. CSP-wLTL
6. EA-CSP-wLTL
*Here wLTL stands for Weighted Logistic Transfer Learning*[3]

![10_20_barplot](/img/[10_20]_barplot.png)
![30_40_barplot](/img/[30_40]_barplot.png)
 
One significant result happened on subject 8 where wLTL perform better than the rest of other methods, this agrees with the study on [3] that wLTL is more pronounced on subject with poor performance.

## Number of Source Data vs Accuracy
![lineplot](/img/lineplot.png)

Classification of left and right hand imagery task.
Different number of target training trials from 20 trials (10 each class) until 40 trials are used to observe the effect it has on accuracy.

## Reference
1. He, H., & Wu, D. (2020). Transfer Learning for Brain-Computer Interfaces: A Euclidean Space Data Alignment Approach. IEEE Transactions on Biomedical Engineering, 67(2), 399–410. https://doi.org/10.1109/TBME.2019.2913914
2. Wu, D., Peng, R., Huang, J., & Zeng, Z. (2020). Transfer Learning for Brain-Computer Interfaces: A Complete Pipeline. 1–9. http://arxiv.org/abs/2007.03746
3. Azab, A. M., Mihaylova, L., Ang, K. K., & Arvaneh, M. (2019). Weighted Transfer Learning for Improving Motor Imagery-Based Brain-Computer Interface. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 27(7), 1352–1359. https://doi.org/10.1109/TNSRE.2019.2923315