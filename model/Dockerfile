FROM python:3.7

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH $PATH:/root/google-cloud-sdk/bin

# Copies the trainer code to the docker image.
COPY . /model

WORKDIR /model

ENV PROJECT_ID=buffer-data

# Set up the entry point to invoke the trainer.
# ENTRYPOINT ["python", "task.py"]
CMD "/bin/bash"
