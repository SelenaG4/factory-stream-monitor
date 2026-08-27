# Serving image: the dashboard only, replaying the committed stream outputs.
# No Spark/Kafka at serve, so the image is small and boots on a free tier.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY dashboard/ dashboard/
COPY data/stream_out/ data/stream_out/
COPY data/model_metrics.json data/model_metrics.json
EXPOSE 8501
CMD streamlit run dashboard/app.py --server.port ${PORT:-8501} --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false
