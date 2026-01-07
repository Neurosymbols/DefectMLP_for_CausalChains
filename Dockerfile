FROM python:3.10-slim

WORKDIR /app

COPY requirements.lock.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        --index-url https://pypi.org/simple \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.lock.txt

COPY . .

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]