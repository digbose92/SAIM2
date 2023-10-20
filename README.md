# SAIM2-BERT
Repository for BERT based domain adaptation experiments using SAIM2

## Data setup

* Download data from here: [**Google Drive**](https://drive.google.com/drive/folders/1jpKHMKkEvXZFWQW9SWys6d3lqM6Ecyog)

* Use the following files:

    * books/books_review_splits_combined.csv

    * dvd/dvd_review_splits_combined.csv

    * electronics/electronics_review_splits_combined.csv

    * kitchen_&_housewares/kitchen_review_splits_combined.csv

## Environment creation

### Source-only transformers training:

* For training the source-only BERT models, use the following environment:

    ```bash
    conda create -n multi_domain_BERT python=3.8
    ```

* Install the required packages using the following command:

    ```bash
    pip install -r requirements.txt
    ```

### Domain adaptation experiments:

* For domain adaptation experiments, use the following environment:
    
    ```bash
    conda create -n domain_adapt_env python=3.7
    ```

* Install tensorflow==2.3.0 using the following command:

    ```bash
    pip install tensorflow-gpu==2.3.0
    ```

