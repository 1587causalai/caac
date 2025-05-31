# Multiclass Classification Results Summary

|   Classes | Data Type   | Model              |   Accuracy |   F1 (Macro) |   AUC (OvR) | Train Time   |
|----------:|:------------|:-------------------|-----------:|-------------:|------------:|:-------------|
|         8 | Clean       | CAAC               |      0.592 |        0.585 |       0.915 | 0.29s        |
|         8 | Clean       | LogisticRegression |      0.708 |        0.71  |       0.936 | 0.01s        |
|         8 | Clean       | RandomForest       |      0.733 |        0.73  |       0.952 | 0.08s        |
|         8 | Clean       | SVM                |      0.658 |        0.658 |       0.953 | 0.02s        |
|         8 | Clean       | MLP                |      0.675 |        0.678 |       0.935 | 0.07s        |
|         8 | 10%         | CAAC               |      0.408 |        0.412 |       0.788 | 0.17s        |
|         8 | 10%         | LogisticRegression |      0.508 |        0.498 |       0.802 | 0.00s        |
|         8 | 10%         | RandomForest       |      0.65  |        0.65  |       0.903 | 0.08s        |
|         8 | 10%         | SVM                |      0.608 |        0.613 |       0.912 | 0.02s        |
|         8 | 10%         | MLP                |      0.517 |        0.515 |       0.809 | 0.07s        |