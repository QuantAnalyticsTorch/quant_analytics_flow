# quant_analytics_flow
Quantitative Analytics using Tensorflow and machine learning


# Install package

pip install -e .
python setup.py bdist_wheel

## Run the tests

pytest --cov-report term --cov=quant_torch tests/ --html=./test-reports/report.html --cov-report=html:./test-reports/coverage --profile

## Upload Python package

python -m twine upload dist/*