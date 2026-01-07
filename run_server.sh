#!/bin/bash

# Build Docker image
docker build -t pcb-defect-api:latest .

# Run Docker container
docker run -d \
  --name pcb-defect-api \
  -p 8000:8000 \
  --restart unless-stopped \
  pcb-defect-api:latest

echo "Container started successfully!"
echo "API available at: http://localhost:8000"
echo "Docs available at: http://localhost:8000/docs"