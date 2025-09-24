#!/usr/bin/env python3
"""
Launch script for Cook Islands Forecast Application
Starts the COG server and opens the forecast web app
"""

import subprocess
import time
import webbrowser
import requests
import sys
import signal
import os

def check_server_health(url, max_retries=30, delay=1):
    """Check if the server is responding"""
    for i in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        
        print(f"Waiting for server... ({i+1}/{max_retries})")
        time.sleep(delay)
    
    return False

def main():
    print("🌊 Starting Cook Islands Wave Forecast Application")
    print("=" * 60)
    
    # Check if server is already running
    try:
        response = requests.get("http://localhost:8000/", timeout=2)
        if response.status_code == 200:
            print("✅ COG Server already running on http://localhost:8000")
            server_process = None
        else:
            raise requests.RequestException("Server not responding correctly")
    except requests.RequestException:
        print("🚀 Starting COG Server...")
        
        # Start the COG server
        server_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print(f"📡 Server process started (PID: {server_process.pid})")
        
        # Wait for server to be ready
        if not check_server_health("http://localhost:8000/"):
            print("❌ Failed to start COG server")
            if server_process:
                server_process.terminate()
            return 1
    
    print("✅ COG Server is running on http://localhost:8000")
    
    # Get the absolute path to the HTML file
    html_path = os.path.join(os.getcwd(), "forecast_app.html")
    app_url = f"file://{html_path}"
    
    print(f"🌐 Opening forecast application: {app_url}")
    
    try:
        # Try to open in browser
        webbrowser.open(app_url)
        print("✅ Forecast application opened in browser")
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print(f"📱 Please open manually: {app_url}")
    
    print("\n" + "🏆" * 20)
    print("🌊 COOK ISLANDS WAVE FORECAST APPLICATION READY!")
    print("🏆" * 20)
    print(f"""
📊 Application Features:
   ✅ Real-time wave height, period, and direction
   ✅ Wind speed and direction forecasts  
   ✅ Coastal inundation depth predictions
   ✅ 9+ day forecast timeline (229 hours)
   ✅ Interactive map with zoom/pan
   ✅ Multiple visualization layers
   ✅ Professional oceanographic data

🎯 Data Source: Pacific Community (SPC) THREDDS Server
🔧 Protocols: OPeNDAP + COG tiles
📡 Server: http://localhost:8000
🌐 App: {app_url}

Press Ctrl+C to stop the server and exit
""")
    
    # Keep the script running and handle cleanup
    try:
        if server_process:
            # Wait for the server process
            server_process.wait()
        else:
            # If server was already running, just wait for user interrupt
            print("Server was already running. Press Ctrl+C to exit launcher.")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        if server_process:
            print("🔄 Stopping COG server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
                print("✅ COG server stopped")
            except subprocess.TimeoutExpired:
                print("⚠️  Force killing server process...")
                server_process.kill()
                server_process.wait()
        
        print("👋 Cook Islands Forecast Application shutdown complete")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)