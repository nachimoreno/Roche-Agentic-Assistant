# Container image for the Roche Scientist Assistant.
#
# Targets Hugging Face Spaces (Docker SDK): HF runs the container as a non-root
# user with UID 1000 and routes HTTP to the port declared as `app_port` in the
# Space README (we use 7860). The same image runs anywhere Docker does.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# onnxruntime (pulled in by fastembed) links libgomp at runtime; it is not in
# the slim base image, so import fails without it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Run as the UID Hugging Face expects, with a writable HOME for the fastembed /
# HF model caches that download on first boot.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

# Dependencies first, so code edits don't bust the install layer.
COPY --chown=user requirements.txt ./
RUN pip install --user -r requirements.txt

# Application code.
COPY --chown=user . .

# Bind to all interfaces on the HF-routed port (overridable via env).
ENV HOST=0.0.0.0 \
    PORT=7860
EXPOSE 7860

CMD ["python", "src/api.py"]
