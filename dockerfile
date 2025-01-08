# Use a Python base image
FROM python:3.9-slim

# Set environment variables to prevent python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (for example, if you need to install any OS libraries)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Install the necessary Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the code and any other necessary files into the container
COPY . /app/

# Expose port 5000 (Optional: if you want to run a web app, adjust as needed)
EXPOSE 8080

# Run the Python script
CMD ["python", "app.ipynb"]
