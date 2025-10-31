# Nueral-Networks

## How to run

to install the required libraries use

```shell
pip install -r requirements.txt
```

open the terminal and run this command

```shell
streamlit run deployment.py
```

## How to train

* enter a name for the model
* choose the two columns
* choose the two classes
* enter the learning rate
* enter number of epochs
* enter the acceptable error
* choose either to add bias or not
* run the model

## How to test

* enter the name of the model you trained to be loaded
* enter both values
* press predict

## notes for the devs

any new added library should be added to *requirment.txt* using

```shell
pip freeze > requirements.txt
```
