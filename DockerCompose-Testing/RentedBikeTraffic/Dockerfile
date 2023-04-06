FROM python:3

COPY ./setup.txt .

RUN pip install -r setup.txt && rm setup.txt

EXPOSE 80

COPY ./app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]