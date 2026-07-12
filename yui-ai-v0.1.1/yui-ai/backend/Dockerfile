FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Processo sem privilégios
RUN useradd --create-home --shell /usr/sbin/nologin yui \
    && chown -R yui:yui /app
USER yui

EXPOSE 8000

# Em produção, rode as migrations antes de subir:
#   alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
