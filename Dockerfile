FROM python:3.8-slim

# this allows this container to be deployed in Google Cloud Run
# remove to run container locally
EXPOSE ${PORT}

WORKDIR /streamlit-app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

COPY marchmadpy/ marchmadpy/
COPY model/ model/
COPY app.py .

ENTRYPOINT python3 -m streamlit run --server.port ${PORT} app.py
