FROM python:3.12-slim

EXPOSE 8501

WORKDIR /app

SHELL ["/bin/bash", "-c"]

# Copy requirements file first to leverage cache
COPY requirements-frontend.txt .

# Install dependencies in a separate layer
RUN python3 -m pip install -r requirements-frontend.txt

# Copy the rest of the code
COPY . .

# USER 65532:65532

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT python3 -m streamlit run frontend/app.py
