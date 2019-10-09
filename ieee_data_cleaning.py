# import streamlit as st
import pandas as pd


def import_data(f):
    test_identity = pd.read_csv("input_ieee/" + f + "_identity.csv")
    test_transaction = pd.read_csv("input_ieee/" + f + "_transaction.csv")
    # sub = pd.read_csv("input_ieee/sample_submission.csv")

    test = pd.merge(test_transaction, test_identity, on="TransactionID", how="left")
    del test_transaction, test_identity
    test.to_csv("input_ieee/" + f + ".csv", index=False)
    pd.read_csv("input_ieee/" + f + ".csv")
    print(test.shape)
    return test


if __name__ == "__main__":

    train = import_data("train")
    del train
    test = import_data("test")
    del test
    # st.write(train.shape)
    # st.write(train)

# test = import_data_test()

# st.write(train.isFraud.value_counts())
# from sklearn.utils import resample

# not_fraud = train[train.isFraud == 0]
# fraud = train[train.isFraud == 1]

# not_fraud_downsampled = resample(
#     not_fraud,
#     replace=False,  # sample without replacement
#     n_samples=400000,  # match minority n
#     random_state=27,
# )  # reproducible results

# # combine minority and downsampled majority
# downsampled = pd.concat([not_fraud_downsampled, fraud])

# # checking counts
# downsampled.isFraud.value_counts()

# train = downsampled.copy()
# del not_fraud_downsampled, downsampled
