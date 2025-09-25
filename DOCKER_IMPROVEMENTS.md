# COG Tiler Docker Improvements

## Immediate Fixes Required

### 1. Create .dockerignore
```
cache/
*.tif
*.tif.lock
__pycache__/
.pytest_cache/
.git/
.vscode/
*.log
nohup.out
venv/
.DS_Store
```

### 2. Fix Port Configuration
- Dockerfile: EXPOSE 8001 (match running service)
- docker-compose: ports: "8084:8001"
- Health check: http://localhost:8001/health

### 3. Fix Health Check
```yaml
healthcheck:
  test: ["CMD", "curl", "-fsS", "http://localhost:8001/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### 4. Add Resource Limits
```yaml
deploy:
  resources:
    limits:
      memory: 2G
      cpus: '1.0'
    reservations:
      memory: 512M
      cpus: '0.5'
```

### 5. Security Improvements
- Add non-root user
- Pin GDAL version
- Add security scanning

### 6. Persistent Cache Strategy
```yaml
volumes:
  - cog_cache:/app/cache
  - /mnt/data/ocean_portal/datasets:/app/datasets:ro
```