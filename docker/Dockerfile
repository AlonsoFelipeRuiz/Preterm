FROM ubuntu:focal

WORKDIR /preterm

RUN apt-get update && apt-get install -y apt-utils python3-pip

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir /models
COPY ./models/kmediods_species_final.sav /models
COPY ./models/kmediods_phylo_final.sav /models
COPY ./models/RandomForest_preterm_final.sav /models
COPY ./models/RandomForest_early_preterm_final.sav /models

COPY main.py .
RUN chmod a+x main.py

ENTRYPOINT ["python3", "main.py"]


