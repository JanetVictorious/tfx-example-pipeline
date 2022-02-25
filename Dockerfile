FROM python:3.8.10

WORKDIR /usr/scr/app

# USER root

RUN mkdir -p data src

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    rm requirements.txt

COPY src ./src

CMD ["python", "src/local_runner.py"]
