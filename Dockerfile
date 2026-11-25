FROM python:3.13-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py ./
COPY assets/examples ./assets/examples
# Weights are not baked in (not committed): mount them at run time,
#   docker run -p 7860:7860 -v "$PWD/weights:/app/weights" tiny-tumor-detector
EXPOSE 7860
CMD ["python", "app.py"]
