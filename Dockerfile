# Production-style image: install deps, train the model at build time, serve with uvicorn.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt     # separate layer -> cached when only code changes
COPY fraud ./fraud
RUN python -m fraud.train                                # bakes models/ into the image (made-up data by default)
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "fraud.api:app", "--host", "0.0.0.0", "--port", "8000"]
