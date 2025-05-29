# Multiclass Classification Results Summary

|   Classes | Data Type   | Model              |   Accuracy |   F1 (Macro) |   AUC (OvR) | Train Time   |
|----------:|:------------|:-------------------|-----------:|-------------:|------------:|:-------------|
|         3 | Clean       | CAAC               |      0.97  |        0.97  |       0.994 | 1.71s        |
|         3 | Clean       | LogisticRegression |      0.893 |        0.893 |       0.968 | 0.01s        |
|         3 | Clean       | RandomForest       |      0.943 |        0.944 |       0.989 | 0.12s        |
|         3 | Clean       | SVM                |      0.97  |        0.97  |       0.994 | 0.02s        |
|         3 | Clean       | MLP                |      0.987 |        0.987 |       0.995 | 0.46s        |
|         3 | 10%         | CAAC               |      0.863 |        0.864 |       0.945 | 1.18s        |
|         3 | 10%         | LogisticRegression |      0.767 |        0.767 |       0.868 | 0.00s        |
|         3 | 10%         | RandomForest       |      0.847 |        0.847 |       0.962 | 0.12s        |
|         3 | 10%         | SVM                |      0.877 |        0.877 |       0.974 | 0.03s        |
|         3 | 10%         | MLP                |      0.887 |        0.887 |       0.929 | 0.46s        |
|         4 | Clean       | CAAC               |      0.845 |        0.845 |       0.951 | 1.36s        |
|         4 | Clean       | LogisticRegression |      0.68  |        0.682 |       0.892 | 0.00s        |
|         4 | Clean       | RandomForest       |      0.835 |        0.835 |       0.963 | 0.17s        |
|         4 | Clean       | SVM                |      0.86  |        0.859 |       0.975 | 0.07s        |
|         4 | Clean       | MLP                |      0.917 |        0.917 |       0.989 | 0.60s        |
|         4 | 10%         | CAAC               |      0.772 |        0.771 |       0.899 | 1.02s        |
|         4 | 10%         | LogisticRegression |      0.618 |        0.617 |       0.815 | 0.00s        |
|         4 | 10%         | RandomForest       |      0.743 |        0.743 |       0.925 | 0.17s        |
|         4 | 10%         | SVM                |      0.775 |        0.774 |       0.949 | 0.08s        |
|         4 | 10%         | MLP                |      0.828 |        0.827 |       0.927 | 0.61s        |
|         5 | Clean       | CAAC               |      0.762 |        0.761 |       0.91  | 3.03s        |
|         5 | Clean       | LogisticRegression |      0.518 |        0.514 |       0.808 | 0.00s        |
|         5 | Clean       | RandomForest       |      0.764 |        0.764 |       0.943 | 0.22s        |
|         5 | Clean       | SVM                |      0.738 |        0.735 |       0.944 | 0.17s        |
|         5 | Clean       | MLP                |      0.884 |        0.884 |       0.984 | 2.44s        |
|         5 | 10%         | CAAC               |      0.638 |        0.64  |       0.846 | 2.11s        |
|         5 | 10%         | LogisticRegression |      0.452 |        0.45  |       0.738 | 0.00s        |
|         5 | 10%         | RandomForest       |      0.712 |        0.71  |       0.909 | 0.22s        |
|         5 | 10%         | SVM                |      0.662 |        0.662 |       0.91  | 0.18s        |
|         5 | 10%         | MLP                |      0.754 |        0.753 |       0.906 | 3.18s        |