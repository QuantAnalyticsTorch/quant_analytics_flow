# quant_analytics_flow
Quantitative Analytics using Tensorflow and machine learning. Full API documentation here

https://quant-analytics-flow.readthedocs.io/


# Install package

pip install -e .
python setup.py bdist_wheel

## Run the tests

pytest --cov-report term --cov=quant_analytics_flow tests/ --html=./test-reports/report.html --cov-report=html:./test-reports/coverage --profile

## Upload Python package

python -m twine upload dist/*

# Sphinx

sphinx-apidoc -o source/ ../quant_analytics_flow
python -m http.server --directory docs/build 9000

# Docker
docker build -t quant_analytics_flow_image .
docker run -d --name quant_analytics_flow_container -p 80:80 quant_analytics_flow_image

# FastAPI
uvicorn main:app --reload