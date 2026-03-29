FROM python:3.10-slim

WORKDIR /code

# Pre-install critical binary requirements for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpng-dev \
    libfreetype6-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Non-root user setup for HF
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
