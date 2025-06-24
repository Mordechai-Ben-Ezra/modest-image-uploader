#!/usr/bin/env bash
# Render מספק ENV VAR בשם PORT
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
