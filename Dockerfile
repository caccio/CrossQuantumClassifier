FROM continuumio/miniconda3

WORKDIR /app

COPY xquantum/ ./xquantum/
COPY test.py ./
COPY driver.py ./
COPY breast-cancer-wisconsin.csv ./
COPY credentials.json ./

RUN wget https://packages.microsoft.com/config/ubuntu/20.10/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
RUN dpkg -i packages-microsoft-prod.deb
RUN apt-get update
RUN apt-get install libgomp1
RUN apt-get install -y dotnet-sdk-3.1
RUN rm packages-microsoft-prod.deb

RUN conda create -n xq -c quantum-engineering qsharp notebook
SHELL ["conda", "run", "-n", "xq", "/bin/bash", "-c"]
RUN pip install qiskit[visualization]
