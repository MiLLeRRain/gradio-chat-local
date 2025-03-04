import os
import sys
import time
import torch
import gradio as gr
from app import create_interface, GPUMonitor, get_auth_credentials
from copilot_proxy import CopilotProxy
import threading

def main():
    # Initialize GPU monitor
    gpu_monitor = GPUMonitor()
    
    # Print GPU information
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available. Running on CPU will be very slow.")
    
    # Initialize Copilot proxy
    copilot_proxy = CopilotProxy()
    
    # Start the Copilot proxy in a separate thread
    print("Starting Copilot proxy server...")
    threading.Thread(target=lambda: copilot_proxy.run(host='127.0.0.1', port=5001), daemon=True).start()
    time.sleep(2)  # Give it time to start
    
    # Create Gradio interface
    print("Creating Gradio interface...")
    interface = create_interface()
    
    # Get authentication credentials
    credentials = get_auth_credentials()
    auth = (credentials['username'], credentials['password'])
    
    # Launch the interface
    print("Launching Gradio interface at http://localhost:7860")
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        auth=auth,
        share=False
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())