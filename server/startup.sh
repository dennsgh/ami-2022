#!/bin/bash

echo "Environment variables : "
env
echo "Mounted disks :"
df -h
echo $(ls /server)
echo "[RUN] gunicorn -b :5000 server.flaskr:app"
gunicorn -b :5000 server.flaskr:app