#!/bin/sh

echo "Starting Scripts"
python making_classifier.py

exec "$@"