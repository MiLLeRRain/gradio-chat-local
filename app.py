import os
import time
import torch
import gradio as gr
import json
import requests
import openai
from flask import Flask, request, redirect, session, url_for
from functools import wraps
from transformers import AutoModelForCausalLM, AutoTokenizer
from duckduckgo_search import DDGS
from copilot_proxy import CopilotProxy

class GPUMonitor:
    def __init__(self):
        self.monitoring = False
        
    def start_monitoring(self):
        self.monitoring = True
        
    def stop_monitoring(self):
        self.monitoring = False
        
    def get_current_stats(self):
        if not torch.cuda.is_available():
            return "GPU not available"
            
        stats = []
        for i in range(torch.cuda.device_count()):
            # Get basic memory stats
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**2    # MB
            max_memory_allocated = torch.cuda.max_memory_allocated(i) / 1024**2  # MB
            
            # Calculate utilization percentage (estimate based on allocated vs reserved)
            utilization_pct = (memory_allocated / memory_reserved * 100) if memory_reserved > 0 else 0
            
            # Try to get temperature if available (requires pynvml)
            temperature = "N/A"
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                temperature = f"{pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU):.1f}°C"
            except (ImportError, Exception):
                # pynvml not available or other error
                pass
                
            gpu_stats = {
                'memory_allocated': memory_allocated,
                'memory_reserved': memory_reserved,
                'max_memory_allocated': max_memory_allocated,
                'utilization_pct': utilization_pct,
                'temperature': temperature
            }
            
            stats.append(
                f"GPU {i} - {torch.cuda.get_device_name(i)}:\n"
                f"  Memory Allocated: {gpu_stats['memory_allocated']:.1f} MB\n"
                f"  Memory Reserved: {gpu_stats['memory_reserved']:.1f} MB\n"
                f"  Max Memory Allocated: {gpu_stats['max_memory_allocated']:.1f} MB\n"
                f"  Memory Utilization: {gpu_stats['utilization_pct']:.1f}%\n"
                f"  Temperature: {gpu_stats['temperature']}"
            )
            
        return "\n\n".join(stats)

# ------------------------------
# 1. Model Loading Configuration
# ------------------------------
MODELS_BASE_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_DIR = os.path.join(MODELS_BASE_DIR, "deepseek-r1", "transformers")  # Default model path
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")    # Local data path
API_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "api_config.json")  # API config path

# Check if directories exist, create if not
os.makedirs(MODELS_BASE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Function to load API configuration
def load_api_config():
    try:
        if os.path.exists(API_CONFIG_PATH):
            print("Loading API config...")
            with open(API_CONFIG_PATH, 'r') as f:
                config = json.load(f)
                return config
        print(f"API config not found at {API_CONFIG_PATH}. Creating default config...")
        return {
            "api_models": ["deepseek-chat", "deepseek-reasoner"],
            "api_keys": {"deepseek": "your_deepseek_api_key"},
            "auth": {"username": "admin", "password": "password"}
        }
    except Exception as e:
        print(f"Error loading API config: {str(e)}")
        return {
            "api_models": ["deepseek-chat", "deepseek-reasoner"],
            "api_keys": {"deepseek": "your_deepseek_api_key"},
            "auth": {"username": "admin", "password": "password"}
        }

# Function to load API model configuration
def load_api_models():
    config = load_api_config()
    return config.get("api_models", [])

# Function to get API key
def get_api_key(provider):
    config = load_api_config()
    print(config)
    return config.get("api_keys", {}).get(provider, "")

# Function to get authentication credentials
def get_auth_credentials():
    config = load_api_config()
    return config.get("auth", {"username": "admin", "password": "password"})

# Function to list available local models
def list_available_local_models():
    if not os.path.exists(MODELS_BASE_DIR):
        return ["No models found. Please place models in the 'models' directory."]
    
    available_models = [d for d in os.listdir(MODELS_BASE_DIR) 
                      if os.path.isdir(os.path.join(MODELS_BASE_DIR, d))]
    
    if not available_models:
        return ["No models found. Please place models in the 'models' directory."]
    
    return available_models

# Function to get models based on mode selection
def get_models_by_mode(use_api_mode):
    if use_api_mode:
        return load_api_models()
    else:
        return list_available_local_models()

# Function to load model with proper error handling
def load_main_model(selected_model, is_api_model=False):
    if is_api_model:
        # This would be replaced with actual API integration code
        return None, None, "API model integration not implemented yet"
    
    try:
        # Construct full model path
        model_path = os.path.join(MODELS_BASE_DIR, selected_model)
        
        # Check if model directory exists
        if not os.path.exists(model_path):
            return None, None, f"Model directory {selected_model} not found"
        
        # Check for version subdirectory (like "1")
        version_dirs = [d for d in os.listdir(model_path) 
                       if os.path.isdir(os.path.join(model_path, d))]
        
        if version_dirs and version_dirs[0].isdigit():
            model_path = os.path.join(model_path, version_dirs[0])
        
        print(f"Loading model from: {model_path}")
        
        # Load model with GPU acceleration
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="cuda",
            low_cpu_mem_usage=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return model, tokenizer, None
    
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        print(error_msg)
        return None, None, error_msg

# Cache for loaded models
main_model_cache = {}

# ------------------------------
# 2. External Search Function
# ------------------------------
def external_api_search(query):
    try:
        context = ""
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)  # Return top 5 results
        
        if results:
            for res in results:
                context += res.get("body", "") + "\n"
        
        return context, None
    except Exception as e:
        error_msg = f"Search error: {str(e)}"
        print(error_msg)
        return "", error_msg

# ------------------------------
# 3. Chunk Generation Function
# ------------------------------
def chunk_generation(prompt, model, tokenizer, max_chunk=512, max_iter=4):
    full_output = ""
    current_prompt = prompt
    
    for _ in range(max_iter):
        inputs = tokenizer(current_prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]
        max_new_tokens = min(max_chunk, 2048 - input_length)  # Adjust based on input length
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,         # Limit new token count
            min_length=10,                         # Set minimum generation length
            do_sample=True,                        # Enable sampling
            temperature=0.7,                       # Control randomness
            top_k=50,                              # Limit sampling range
            top_p=0.95,                            # Nucleus sampling
            num_beams=3,                           # Beam search
            early_stopping=True,                   # Enable early stopping
            repetition_penalty=1.2,                # Reduce repetition
            no_repeat_ngram_size=3,                # Prevent n-gram repetition
            use_cache=True,                        # Enable cache for speed
            pad_token_id=model.config.eos_token_id,  # Use EOS token as padding
            output_scores=False,
            output_attentions=False,
            output_hidden_states=False,
            num_return_sequences=1
        )
        
        chunk = tokenizer.decode(outputs[0], skip_special_tokens=True)
        full_output += chunk
        
        if model.config.eos_token_id in outputs[0]:
            break
            
        current_prompt = full_output
    
    return full_output

# ------------------------------
# 4. Main Generation Function
# ------------------------------
def hybrid_generation(query, model_choice, use_external_search, use_chunk_generation, use_api_mode, system_prompt=""):
    # Check if model choice is valid
    if model_choice.startswith("No models found"):
        return "Error: No models available. Please add models to the 'models' directory.", ""
    
    # Handle API model case
    if use_api_mode:
        # Get API key for DeepSeek
        api_key = get_api_key("deepseek")
        if not api_key or api_key == "your_deepseek_api_key":
            return "Error: API key not configured. Please update the api_config.json file with your API key.", ""
        
        try:
            # Start timing
            start_time = time.time()
            
            # Setup for OpenAI-compatible API call to DeepSeek
            import openai
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com"
            )
            
            # Prepare messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": query})
            
            # Make API request to DeepSeek
            print("\n[API Request to DeepSeek]")
            print("Messages:")
            print(json.dumps(messages, indent=2))
            
            response = client.chat.completions.create(
                model=model_choice,
                messages=messages,
                temperature=0.7,
                max_tokens=800,
                stream=False
            )
            
            print("\n[API Response from DeepSeek]")
            print(json.dumps(response.model_dump(), indent=2))
            
            # Extract response content
            reply = response.choices[0].message.content
            token_count = response.usage.total_tokens
            
            # Calculate statistics
            elapsed_time = time.time() - start_time
            stats = (
                f"API 生成统计:\n"
                f"- 模型: {model_choice}\n"
                f"- Token 数量: {token_count}\n"
                f"- 生成耗时: {elapsed_time:.2f} 秒\n"
                f"- 生成速度: {token_count/elapsed_time:.2f} tokens/second"
            )
            
            return "最终回答：" + reply, stats
                
        except Exception as e:
            error_msg = f"API Error: {str(e)}"
            return error_msg, ""

    # Load selected model (using cache)
    if model_choice not in main_model_cache:
        model, tokenizer, error = load_main_model(model_choice, is_api_model=use_api_mode)
        
        if error:
            return f"Error: {error}", ""
            
        main_model_cache[model_choice] = (model, tokenizer)
    else:
        model, tokenizer = main_model_cache[model_choice]
    
    # Get search context if enabled
    search_context = ""
    if use_external_search:
        search_context, error = external_api_search(query)
        if error:
            return f"Search error: {error}\n\nProceeding with generation without search results.", ""
    
    # Combine query with search context
    combined_query = ""
    if search_context:
        combined_query += f"搜索上下文: {search_context}\n"
    combined_query += f"原问题: {query}"
    
    # Start timing
    start_time = time.time()
    
    try:
        # Generate response using selected method
        if use_chunk_generation:
            reply = chunk_generation(combined_query, model, tokenizer, max_chunk=512, max_iter=4)
            # For token count estimation in chunk mode
            token_count = len(tokenizer.encode(reply))
        else:
            inputs = tokenizer(combined_query, return_tensors="pt").to(model.device)
            input_length = inputs.input_ids.shape[1]
            max_new_tokens = min(800, 2048 - input_length)  # Adjust based on input length
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,         # Limit new token count
                min_length=10,                         # Set minimum generation length
                do_sample=True,                        # Enable sampling
                temperature=0.7,                       # Control randomness
                top_k=50,                              # Limit sampling range
                top_p=0.95,                            # Nucleus sampling
                num_beams=3,                           # Beam search
                early_stopping=True,                   # Enable early stopping
                repetition_penalty=1.2,                # Reduce repetition
                no_repeat_ngram_size=3,                # Prevent n-gram repetition
                use_cache=True,                        # Enable cache for speed
                pad_token_id=model.config.eos_token_id,  # Use EOS token as padding
                output_scores=False,
                output_attentions=False,
                output_hidden_states=False,
                num_return_sequences=1
            )
            
            reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
            token_count = outputs.shape[-1]
        
        # Calculate statistics
        elapsed_time = time.time() - start_time
        # Get final GPU stats
        final_gpu_stats = gpu_monitor.get_current_stats()
        stats = (
            f"生成统计:\n"
            f"- Token 数量: {token_count}\n"
            f"- 生成耗时: {elapsed_time:.2f} 秒\n"
            f"- 生成速度: {token_count/elapsed_time:.2f} tokens/second\n\n"
            f"GPU 状态:\n{final_gpu_stats}"
        )
        
        # Clean up GPU memory and stop monitoring
        torch.cuda.empty_cache()
        gpu_monitor.stop_monitoring()
        
        return "最终回答：" + reply, stats
    
    except Exception as e:
        error_msg = f"Generation error: {str(e)}"
        print(error_msg)
        return error_msg, ""

# ------------------------------
# 5. Gradio Interface
# ------------------------------
# Function to test API connection
def test_api_connection(provider="deepseek"):
    api_key = get_api_key(provider)
    if not api_key or api_key == "your_deepseek_api_key":
        return f"Error: {provider} API key not configured. Please update the api_config.json file with your API key."
    
    try:
        # Use OpenAI client with custom settings
        client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=5.0,  # Set a reasonable timeout
            max_retries=0  # Disable automatic retries
        )
        
        # Prepare request payload
        request_payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1,
            "timeout": 5.0,
            "stream": False
        }
        print("\n[API Request Payload]")
        print(json.dumps(request_payload, indent=2))
        
        # Record start time
        start_time = time.time()
        
        # Make a minimal test request
        response = client.chat.completions.create(**request_payload)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Log response details
        print("\n[API Response]")
        print(f"Response Time: {response_time:.2f} seconds")
        print("Response Content:")
        print(json.dumps(response.model_dump(), indent=2))
        
        return f"✅ Successfully connected to {provider.capitalize()} API! Available models include deepseek-chat (V3) and deepseek-reasoner (R1)."
            
    except openai.APITimeoutError:
        return f"❌ Connection timeout: The {provider} API is taking too long to respond. Please try again."
    except Exception as e:
        return f"❌ Error testing API connection: {str(e)}"

# Function to test Copilot proxy connection - replaced with notification
def test_copilot_connection():
    return "⚠️ Copilot proxy is currently under development"

# Function to toggle Copilot proxy - replaced with notification
def toggle_copilot_proxy(enable):
    return "⚠️ Copilot proxy is currently under development", gr.update(visible=True)

def create_interface():
    # Initialize with local models
    available_local_models = list_available_local_models()
    available_api_models = load_api_models()
    
    with gr.Blocks(title="混合对话生成系统") as interface:
        gr.Markdown("# 混合对话生成系统")
        gr.Markdown("选择模型类型、主模型、是否启用联网搜索和分块生成策略，系统将整合搜索上下文和动态生成参数生成最终回复，并显示生成统计信息。")
        
        # State for persisting mode selection
        use_api_mode_state = gr.State(value=False)
        
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(lines=2, placeholder="请输入你的问题...")
                
                # Add model type toggle
                with gr.Row():
                    use_api_mode = gr.Checkbox(label="使用API模型", value=False)
                    # Add API test button
                    test_api_btn = gr.Button("测试 API 连接", visible=False)
                
                # Model selection dropdown that updates based on toggle
                model_choice = gr.Dropdown(
                    choices=available_local_models, 
                    label="选择主模型", 
                    value=available_local_models[0] if available_local_models else None
                )
                
                # Add system prompt for API models
                system_prompt = gr.Textbox(
                    lines=2, 
                    placeholder="输入系统提示（可选，仅API模式有效）...",
                    label="系统提示（System Prompt）",
                    visible=False
                )
                
                with gr.Row():
                    use_search = gr.Checkbox(label="启用 DuckDuckGo 联网搜索", value=True)
                    use_chunk = gr.Checkbox(label="启用分块生成策略", value=False)
                
                # Modified Copilot proxy toggle and test button
                with gr.Row():
                    use_copilot = gr.Checkbox(label="启用 Copilot 代理（开发中）", value=False, interactive=False)
                    test_copilot_btn = gr.Button("测试 Copilot 连接")
                
                submit_btn = gr.Button("生成回答")
            
            with gr.Column(scale=1):
                gpu_stats = gr.Textbox(label="实时 GPU 状态", value="等待生成...", interactive=False)
                # Add API test result display
                api_test_result = gr.Textbox(label="API 连接测试结果", visible=False)
                # Add Copilot test result display
                copilot_test_result = gr.Textbox(label="Copilot 连接测试结果", visible=False)
        
        response = gr.Textbox(label="生成回答")
        stats = gr.Textbox(label="生成统计信息")
        
        def update_gpu_stats():
            return gpu_monitor.get_current_stats()
        
        # Function to update model choices based on toggle
        def update_model_choices(use_api):
            use_api_mode_state.value = use_api
            # Show/hide API test button and system prompt based on mode
            if use_api:
                return gr.Dropdown(choices=available_api_models, value=available_api_models[0] if available_api_models else None), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
            else:
                return gr.Dropdown(choices=available_local_models, value=available_local_models[0] if available_local_models else None), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
        
        # Connect toggle to update model choices and API test button visibility
        use_api_mode.change(
            fn=update_model_choices,
            inputs=[use_api_mode],
            outputs=[model_choice, test_api_btn, api_test_result, system_prompt]
        )
        
        # Connect API test button
        test_api_btn.click(
            fn=test_api_connection,
            inputs=[],
            outputs=[api_test_result]
        )
        
        # Connect Copilot toggle and test button - modified to show notification
        use_copilot.change(
            fn=lambda x: "⚠️ Copilot proxy is currently under development",
            inputs=[use_copilot],
            outputs=[copilot_test_result]
        )
        
        test_copilot_btn.click(
            fn=lambda: "⚠️ Copilot proxy is currently under development",
            inputs=[],
            outputs=[copilot_test_result]
        )
        
        submit_btn.click(
            fn=hybrid_generation,
            inputs=[query_input, model_choice, use_search, use_chunk, use_api_mode, system_prompt],
            outputs=[response, stats]
        )
        
        # Update GPU stats every 5 seconds using proper event handling
        gpu_monitor.start_monitoring()
        
        # Create a refresh button that will be automatically clicked
        refresh_btn = gr.Button("Refresh GPU Stats", visible=False)
        refresh_btn.click(
            fn=update_gpu_stats,
            inputs=[],
            outputs=[gpu_stats]
        )
        
        # Initial update of GPU stats
        gpu_stats.value = update_gpu_stats()
        
        # Set up a timer in the interface load event to periodically refresh GPU stats
        # Using gr.HTML for JavaScript execution instead of the unsupported _js parameter
        gr.HTML("""
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                setInterval(function() {
                    const buttons = Array.from(document.querySelectorAll('button'));
                    const refreshBtn = buttons.find(btn => btn.textContent.includes('Refresh GPU Stats'));
                    if (refreshBtn) refreshBtn.click();
                }, 5000);
            });
        </script>
        """, visible=False)
    
    return interface

# ------------------------------
# 6. Authentication System
# ------------------------------
# Create Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Secret key for session management

# Authentication decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        credentials = get_auth_credentials()
        if (request.form['username'] == credentials['username'] and
                request.form['password'] == credentials['password']):
            session['logged_in'] = True
            return redirect(url_for('index'))
        else:
            error = 'Invalid credentials. Please try again.'
    
    # Simple login form
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Login</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #f5f5f5; }
            .login-container { background-color: white; padding: 2rem; border-radius: 5px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); width: 300px; }
            h2 { text-align: center; margin-bottom: 1.5rem; color: #333; }
            .form-group { margin-bottom: 1rem; }
            label { display: block; margin-bottom: 0.5rem; color: #555; }
            input[type="text"], input[type="password"] { width: 100%; padding: 0.5rem; border: 1px solid #ddd; border-radius: 3px; }
            button { width: 100%; padding: 0.75rem; background-color: #4CAF50; color: white; border: none; border-radius: 3px; cursor: pointer; font-size: 1rem; }
            button:hover { background-color: #45a049; }
            .error { color: red; margin-bottom: 1rem; }
        </style>
    </head>
    <body>
        <div class="login-container">
            <h2>Login</h2>
            ''' + (f'<p class="error">{error}</p>' if error else '') + '''
            <form method="post">
                <div class="form-group">
                    <label for="username">Username:</label>
                    <input type="text" id="username" name="username" required>
                </div>
                <div class="form-group">
                    <label for="password">Password:</label>
                    <input type="password" id="password" name="password" required>
                </div>
                <button type="submit">Login</button>
            </form>
        </div>
    </body>
    </html>
    '''

# Logout route
@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))

# Main route that serves the Gradio app
@app.route('/')
@login_required
def index():
    return redirect('/gradio')

# ------------------------------
# 7. Main Application
# ------------------------------
if __name__ == "__main__":
    # Initialize GPU monitor
    gpu_monitor = GPUMonitor()
    # Print GPU information
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available. Running on CPU will be very slow.")
    
    # Commented out: Initialize Copilot proxy (but don't start it yet)
    # copilot_proxy = CopilotProxy()
    
    # Create Gradio interface
    interface = create_interface()
    
    # Launch Gradio with authentication
    credentials = get_auth_credentials()
    auth = (credentials['username'], credentials['password'])
    
    # Launch the interface directly
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        auth=auth,
        share=False
    )