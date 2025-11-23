# 1. Start with a lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file first (for better caching)
COPY requirements.txt .

# 4. Install dependencies
# --no-cache-dir keeps the image small
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code and artifacts
# This copies main.py, .json files, .joblib files, etc.
COPY . .

# 6. Expose the port the app runs on
EXPOSE 8000

# 7. Define the command to run the app
# host 0.0.0.0 is crucial for Docker networking
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]