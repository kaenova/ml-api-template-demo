FROM python:3.10.3-slim-buster

WORKDIR /workspace

# Karena kita tidak commit file model dan hanya akan di download, maka
# kita harus download secara manual
RUN apt-get update -y
RUN apt-get install -y wget

# Text Model (saved model that is still in zip)
RUN wget https://huggingface.co/kaenova/simple-model-demo/resolve/main/text.zip?download=true
RUN unzip text.zip
# Delete unused model zip file (already extracted)
RUN rm text.zip

# Vision model (h5 model)
RUN wget https://huggingface.co/kaenova/simple-model-demo/resolve/main/vision.h5?download=true

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

ENV HOST 0.0.0.0

EXPOSE 8080

CMD ["python", "main.py"]
