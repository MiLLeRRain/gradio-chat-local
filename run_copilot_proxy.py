import sys
import os
import time
import argparse
from copilot_proxy import CopilotProxy

def main():
    parser = argparse.ArgumentParser(description="Run the Copilot Proxy Server")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address to bind to')
    parser.add_argument('--port', type=int, default=5001, help='Port to run the proxy on')
    parser.add_argument('--config', type=str, default='copilot_config.json', help='Path to config file')
    
    args = parser.parse_args()
    
    proxy = CopilotProxy(config_path=args.config)
    try:
        proxy.run(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("Shutting down Copilot Proxy Server...")
        proxy.stop()

if __name__ == "__main__":
    main()