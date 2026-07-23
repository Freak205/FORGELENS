FROM python:3.14-slim
WORKDIR /opt/forgelens
COPY . .
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.12.1 torchvision==0.27.1 && \
    python -m pip install --no-cache-dir huggingface-hub==1.24.0 numpy==2.3.2 pydantic==2.13.4 PyYAML==6.0.3 structlog==24.4.0
ENV PYTHONPATH=/opt/forgelens/src
EXPOSE 7860
CMD ["python", "demo/server.py"]
