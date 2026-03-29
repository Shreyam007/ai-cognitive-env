FROM python:3.10

# Create user with UID 1000 exactly as HF needs
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Only copy requirements for cache-efficient installs
COPY --chown=user requirements.txt $HOME/app/
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy literally all code, including inference.py
COPY --chown=user . $HOME/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
