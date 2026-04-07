FROM python:3.10-slim

WORKDIR /code

# Pre-install requirements for fast caching
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Non-root user setup for HF (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Ensure we use the correct module:app syntax
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
