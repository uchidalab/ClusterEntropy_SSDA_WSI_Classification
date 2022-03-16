# ST
The codes for experiment S+T (use labeled source abd single labeled target for train data) \
Single labeled target WSI is selected by cluster entropy.

## How to use
- **Train model**: \
    Use train.py
- **Test the trained model**: \
    Use test.py
- **Make segmentation (prediction) map**: \
    Use predict_faster.py. Execution time of predict.py is much slower than predict_faster.py. So reccomend to use predict_faster.py.
