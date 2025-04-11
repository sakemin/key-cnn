FROM python:3.7-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
# Fix protobuf compatibility issue with TensorFlow 1.15.2
RUN pip install --no-cache-dir protobuf==3.20.0
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Set the default command
ENTRYPOINT ["python", "predict.py"]
CMD ["audio_file.wav", "--model", "ensemble", "--alpha", "0.5"] 