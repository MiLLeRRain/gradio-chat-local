import sys
import os
import time
from copilot_proxy import CopilotProxy

def main():
    print("Starting Copilot Proxy Server...")
    print("Press Ctrl+C to stop the server")
    
    # Initialize the proxy
    proxy = CopilotProxy()
    
    try:
        # Run the proxy server
        proxy.run(host='127.0.0.1', port=5001)
    except KeyboardInterrupt:
        print("\nStopping Copilot Proxy Server...")
        proxy.stop()
        print("Server stopped")
    except Exception as e:
        print(f"Error running Copilot Proxy: {str(e)}")
        proxy.stop()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())