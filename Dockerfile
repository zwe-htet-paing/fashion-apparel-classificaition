FROM python:3.12

RUN pip install --upgrade pip
RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["/model/" , "./model/"]
COPY ["predict.py", "./"]

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "predict.py"]