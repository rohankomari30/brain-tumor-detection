FROM python:3.10-slim

WORKDIR /app

# System dependency needed by opencv-python (used in the CLAHE experiment's
# preprocessing function, imported indirectly if present in the codebase)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY . .

EXPOSE 7860

# --timeout 120: the app runs two models plus Grad-CAM and occlusion
# sensitivity per request, which takes longer than gunicorn's 30s default.
# Shell form (not exec form) so $PORT gets expanded at container start -
# Render assigns this dynamically; falls back to 7860 if unset.
CMD gunicorn --bind 0.0.0.0:${PORT:-7860} --timeout 120 --workers 1 app:app
