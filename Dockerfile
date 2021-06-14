FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY requirements.txt /app/requirements.txt
COPY /dist/* /app/

WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install quant_analytics_flow*

COPY ./app /app


