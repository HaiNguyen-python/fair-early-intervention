# Pinned to Debian bookworm: plain -slim tags now move to trixie.
FROM python:3.12-slim-bookworm

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Train on synthetic data at build time so the image serves out of the box.
# For real OULAD data, mount the CSVs and run `fair-intervention train --source oulad`.
ENV FAIR_INTERVENTION_ARTIFACTS=/app/artifacts
RUN fair-intervention train --artifacts /app/artifacts

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["fair-intervention", "serve", "--port", "8000"]
