# Multiclass Classification Results Summary

|   Classes | Data Type   | Model              |   Accuracy |   F1 (Macro) |   AUC (OvR) | Train Time   |
|----------:|:------------|:-------------------|-----------:|-------------:|------------:|:-------------|
|         3 | Clean       | CAAC               |      0.85  |        0.849 |       0.954 | 0.28s        |
|         3 | Clean       | LogisticRegression |      0.9   |        0.9   |       0.972 | 0.01s        |
|         3 | Clean       | RandomForest       |      0.875 |        0.875 |       0.971 | 0.07s        |
|         3 | Clean       | SVM                |      0.867 |        0.867 |       0.974 | 0.01s        |
|         3 | Clean       | MLP                |      0.883 |        0.883 |       0.976 | 0.06s        |
|         3 | 10%         | CAAC               |      0.717 |        0.71  |       0.882 | 0.38s        |
|         3 | 10%         | LogisticRegression |      0.808 |        0.809 |       0.882 | 0.00s        |
|         3 | 10%         | RandomForest       |      0.825 |        0.825 |       0.939 | 0.07s        |
|         3 | 10%         | SVM                |      0.817 |        0.818 |       0.952 | 0.01s        |
|         3 | 10%         | MLP                |      0.833 |        0.834 |       0.891 | 0.06s        |