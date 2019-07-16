import numpy as np
from sklearn.model_selection import ShuffleSplit
import pandas as pd

niter = 100
nsample = 1536
cols = [("train{}".format(i), "test{}".format(i)) for i in range(100)]

df = pd.DataFrame(
     data=np.zeros((int(nsample*.8), int(niter*2))),
     columns=[k for i in cols for k in i]
)

X, y = np.random.random((nsample, 1)), np.random.random(nsample)
cv = ShuffleSplit(n_splits=niter, test_size=.2)


def pad(data, l, pad_val=123456789):
    padded_data = np.repeat(pad_val, l)
    padded_data[:len(data)] = data
    return padded_data

for idx, (train, test) in enumerate(cv.split(X, y)):
    df["train{}".format(idx)] = train
    df["test{}".format(idx)] = pad(test, int(nsample*.8))
    
df.to_csv("shuffle_split_cv.csv")
