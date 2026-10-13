FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml requirements.txt README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -e ".[dashboard]"
COPY app ./app
COPY scripts ./scripts
COPY .streamlit ./.streamlit
# results/ is mounted at run time. If empty, a SIMULATED demo is generated so the container
# starts; it is labelled SIMULATED everywhere. Real results: mount a results/ from a real run.
ENV RESULTS_DIR=/app/results
EXPOSE 8000 8501
COPY docker-entrypoint.sh /usr/local/bin/entrypoint
RUN chmod +x /usr/local/bin/entrypoint
ENTRYPOINT ["entrypoint"]
CMD ["api"]
