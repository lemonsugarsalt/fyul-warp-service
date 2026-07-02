FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir flask pillow numpy gunicorn
COPY warp_service.py .
EXPOSE 8080
CMD ["/bin/sh", "-c", "python warp_service.py"]
