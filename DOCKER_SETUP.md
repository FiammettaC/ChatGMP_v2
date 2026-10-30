# Docker Setup Guide for ChatGMP

## Prerequisites
- Docker must be installed on your system
- Docker Compose (usually comes with Docker Desktop)

## Quick Start

### Using Docker Compose (Recommended)

1. **Copy the environment template:**
   ```bash
   copy {dot}env.example .env
   ```
   Or on Linux/Mac:
   ```bash
   cp {dot}env.example .env
   ```

2. **Edit the `.env` file and add your API keys:**
   ```
   LECHAT_API_KEY=your_actual_mistral_api_key
   FLASK_SECRET_KEY=your_secure_secret_key
   ```

3. **Build and run the container:**
   ```bash
   docker-compose up --build
   ```

4. **Access the application:**
   - Open your browser to `http://localhost:5000`

### Manual Docker Build and Run

1. **Build the Docker image:**
   ```bash
   docker build -t chatgmp:latest .
   ```

2. **Run the container:**
   ```bash
   docker run -d \
     -p 5000:5000 \
     -v %cd%/data:/app/data \
     -v %cd%/static:/app/static \
     -v %cd%/templates:/app/templates \
     --name chatgmp \
     chatgmp:latest
   ```

3. **Access the application:**
   - Open your browser to `http://localhost:5000`

## Useful Docker Commands

- **View logs:**
  ```bash
  docker-compose logs -f
  ```

- **Stop the container:**
  ```bash
  docker-compose down
  ```

- **Rebuild the image:**
  ```bash
  docker-compose up --build
  ```

- **Remove everything (including volume data):**
  ```bash
  docker-compose down -v
  ```

## Security Notes

- API keys are stored in the `.env` file (not version controlled)
- Never commit `.env` to version control
- The `.dockerignore` file prevents unnecessary files from being included in the image
- The `{dot}env.example` file shows what variables are needed (safe to commit)

## Troubleshooting

- **Port already in use:** Change the port mapping in `docker-compose.yml` (e.g., `"8000:5000"`)
- **Permission denied:** On Linux, run with `sudo docker-compose up`
- **Volume permission issues:** Check file ownership in your data/static/templates directories

## Production Deployment

For production, consider:
1. Using a production WSGI server like Gunicorn instead of Flask's development server
2. Setting `FLASK_ENV=production` and `FLASK_DEBUG=0`
3. Using a reverse proxy (Nginx) in front of the app
4. Using proper secret management (not .env files)
5. Adding resource limits to the container
6. Using a multi-stage Dockerfile to reduce image size
