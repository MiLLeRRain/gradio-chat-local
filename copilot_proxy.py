import os
import json
import subprocess
import requests
import time
import threading
import logging
from flask import Flask, request, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('copilot_proxy')

class CopilotProxy:
    def __init__(self, config_path="copilot_config.json"):
        self.config_path = config_path
        self.config = self.load_config()
        self.server_process = None
        self.server_url = f"http://localhost:{self.config.get('port', 5000)}"
        self.app = Flask(__name__)
        self.setup_routes()
        
    def load_config(self):
        """Load configuration from JSON file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
            else:
                # Default configuration
                default_config = {
                    "port": 5000,
                    "server_path": "@github/copilot-language-server",
                    "timeout": 30
                }
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                return default_config
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            return {"port": 5000, "server_path": "@github/copilot-language-server", "timeout": 30}
    
    def start_server(self):
        """Start the Copilot Language Server"""
        try:
            # Check if server is already running
            try:
                response = requests.get(f"{self.server_url}/health", timeout=2)
                if response.status_code == 200:
                    logger.info("Copilot server is already running")
                    return True
            except requests.exceptions.RequestException:
                pass  # Server not running, continue with startup
                
            logger.info("Starting Copilot Language Server...")
            server_path = self.config.get('server_path', '@github/copilot-language-server')
            
            # On Windows, we need to use shell=True for npm and npx commands
            use_shell = os.name == 'nt'  # True on Windows
            
            # Check if the Copilot language server package is installed
            try:
                # Try to install the package if it's not already installed
                install_cmd = "npm install -g " + server_path if use_shell else ["npm", "install", "-g", server_path]
                logger.info(f"Installing Copilot language server: {install_cmd if use_shell else ' '.join(install_cmd)}")
                subprocess.run(install_cmd, check=True, capture_output=True, text=True, shell=use_shell)
                logger.info("Copilot language server installed successfully")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to install Copilot language server: {e.stderr}")
                logger.info("Continuing with npx to run the server...")
            except Exception as e:
                logger.warning(f"Error during package installation: {str(e)}")
                logger.info("Continuing with npx to run the server...")
            
            # Start the server as a background process
            cmd = "npx --yes " + server_path + " --port " + str(self.config.get('port', 5000)) if use_shell else ["npx", "--yes", server_path, "--port", str(self.config.get('port', 5000))]
            logger.info(f"Running command: {cmd if use_shell else ' '.join(cmd)}")
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=use_shell
            )
            
            # Wait for server to start
            timeout = self.config.get('timeout', 30)
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = requests.get(f"{self.server_url}/health", timeout=2)
                    if response.status_code == 200:
                        logger.info("Copilot server started successfully")
                        return True
                except requests.exceptions.RequestException:
                    time.sleep(1)
                    
                    # Check if process has terminated with error
                    if self.server_process.poll() is not None:
                        stderr_output = self.server_process.stderr.read()
                        logger.error(f"Copilot server process terminated with error: {stderr_output}")
                        return False
                    
            logger.error("Failed to start Copilot server within timeout")
            return False
        except Exception as e:
            logger.error(f"Error starting Copilot server: {str(e)}")
            return False
    
    def setup_routes(self):
        """Set up Flask routes for the proxy"""
        @self.app.route('/health', methods=['GET'])
        def health_check():
            try:
                response = requests.get(f"{self.server_url}/health", timeout=5)
                return jsonify({"status": "ok", "server": response.json()}), response.status_code
            except requests.exceptions.RequestException as e:
                return jsonify({"status": "error", "message": str(e)}), 500
        
        @self.app.route('/v1/completions', methods=['POST'])
        def completions():
            try:
                # Forward the request to the Copilot server
                response = requests.post(
                    f"{self.server_url}/v1/completions",
                    json=request.json,
                    headers=request.headers,
                    timeout=30
                )
                return jsonify(response.json()), response.status_code
            except requests.exceptions.RequestException as e:
                return jsonify({"status": "error", "message": str(e)}), 500
    
    def run(self, host='0.0.0.0', port=None):
        """Run the proxy server"""
        if port is None:
            port = self.config.get('proxy_port', 5001)
            
        # Start the Copilot server in a separate thread
        threading.Thread(target=self.start_server, daemon=True).start()
        
        # Run the Flask app
        self.app.run(host=host, port=port, debug=False)
    
    def stop(self):
        """Stop the Copilot server"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process = None
            logger.info("Copilot server stopped")

# Main entry point
if __name__ == "__main__":
    proxy = CopilotProxy()
    proxy.run()