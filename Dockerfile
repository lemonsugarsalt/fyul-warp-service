FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir flask pillow numpy gunicorn
COPY warp_service.py .
EXPOSE 8080
CMD gunicorn --bind 0.0.0.0:$PORT warp_service:app
