#!/bin/sh

echo "Starting Scripts"
python scrapping_amazon.py

exec "$@"