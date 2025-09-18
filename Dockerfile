# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app's code to the working directory
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the app when the container launches
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]