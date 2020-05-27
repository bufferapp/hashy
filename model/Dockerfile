FROM python:3.7

COPY requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN curl -sSL https://sdk.cloud.google.com | bash
ENV PATH $PATH:/root/google-cloud-sdk/bin

COPY . /model

WORKDIR /model

CMD ["bash", "scripts/train.sh", "buffer-data.dbt_michael.predict_hashtags_posts" ]
