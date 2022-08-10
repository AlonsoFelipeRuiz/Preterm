FROM python:3.8-alpine

WORKDIR /preterm

RUN apk update && apk add make automake gcc g++

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .
RUN chmod a+x main.py

ENTRYPOINT ["python", "main.py"]


