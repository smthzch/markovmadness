# March Madness App

This repo contains code to:

1. Get NCAA basketball games scores: `python get_data.py --help`
2. Fit a Markov model on score: `python train.py --help`
3. Run an app to make matchup predictions: `streamlit run app.py`

A dockerfile is also included in order to package the streamlit app for deployment to Google Cloud Run.

## Setup

This was tested with python 3.8

```shell
pip install -r requirements.txt
```
