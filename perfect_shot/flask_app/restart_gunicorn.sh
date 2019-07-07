#!/bin/bash
pkill gunicorn
gunicorn3 app:app --timeout 600 -D
