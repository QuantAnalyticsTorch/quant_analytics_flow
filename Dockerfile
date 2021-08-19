FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY requirements.txt /app/requirements.txt
COPY /dist/* /app/

WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install quant_analytics_flow*

COPY ./app /app

CMD jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root
