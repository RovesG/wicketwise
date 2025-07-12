# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Add a non-root user for security
RUN useradd -m --uid 1001 --gid 0 appuser
COPY . /app
RUN chown -R 0:0 /app

# Switch to the non-root user
USER appuser
WORKDIR /home/appuser/app

# Install dependencies
# First, install torch and its related packages using the extra index URL
# to ensure compatibility for torch-geometric.
# This is for a CPU-only build. For GPU, the torch version and index URL
# would need to be adjusted.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
        torch-scatter \
        torch-sparse \
        torch-cluster \
        torch-spline-conv \
        --index-url https://data.pyg.org/whl/torch-2.5.1+cpu.html

# Install the remaining dependencies from requirements.txt
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Set the default command to run the Streamlit app
CMD ["streamlit", "run", "ui_streamlit.py"] 