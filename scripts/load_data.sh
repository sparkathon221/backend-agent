#!/bin/bash
curl -L -o ./amazon-products-dataset.zip\
  https://www.kaggle.com/api/v1/datasets/download/lokeshparab/amazon-products-dataset

unzip ./amazon-products-dataset.zip -d ./amazon-products-dataset
rm ./amazon-products-dataset.zip

