# ObjectDetection

## Local running
streamlit run --server.port 8090 objectdetection.py

## Docker
# Build image
docker build -f Dockerfile -t objectdetection:latest .

# Run image locally
sudo docker run --it -p8090:8080 --rm objectdetection:latest
