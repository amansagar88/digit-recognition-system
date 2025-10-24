    # Use the standard Python 3.10 runtime
    FROM python:3.10 
    
    # Set the working directory in the container
    WORKDIR /app
    
    # Copy the requirements file first for layer caching
    COPY requirements.txt .
    
    # Update package lists and install necessary system dependencies for OpenCV
    # Separated update and install for clarity
    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgtk2.0-dev && \
        # Clean up apt lists to reduce image size
        rm -rf /var/lib/apt/lists/*
    
    # Install Python dependencies
    RUN pip install --no-cache-dir -r requirements.txt
    
    # Copy the rest of the application code into the container
    COPY . .
    
    # Expose the port Gunicorn will run on (HF standard)
    EXPOSE 7860 
    
    # Define environment variable (optional)
    ENV NAME DigitRecognizer
    
    # Run app.py using Gunicorn
    CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
    

