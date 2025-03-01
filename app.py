import os
import time
import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from duckduckgo_search import DDGS

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
            gpu_stats = {
                'memory_allocated': torch.cuda.memory_allocated(i) / 1024**2,  # MB
                'memory_reserved': torch.cuda.memory_reserved(i) / 1024**2,      # MB
                'max_memory_allocated': torch.cuda.max_memory_allocated(i) / 1024**2  # MB
            }
            
            stats.append(
                f"GPU {i} - {torch.cuda.get_device_name(i)}:\n"
                f"  Memory Allocated: {gpu_stats['memory_allocated']:.1f} MB\n"
                f"  Memory Reserved: {gpu_stats['memory_reserved']:.1f} MB\n"
                f"  Max Memory Allocated: {gpu_stats['max_memory_allocated']:.1f} MB"
            )
            
        return "\n\n".join(stats)

# ------------------------------
# 1. Model Loading Configuration
# ------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "deepseek-r1", "transformers")  # Updated model path
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")    # Local data path

# Check if models directory exists, create if not
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Function to list available models
def list_available_models():
    if not os.path.exists(MODEL_DIR):
        return ["No models found. Please place models in the 'models' directory."]
    
    available_models = [d for d in os.listdir(MODEL_DIR) 
                      if os.path.isdir(os.path.join(MODEL_DIR, d))]
    
    if not available_models:
        return ["No models found. Please place models in the 'models' directory."]
    
    return available_models

# Function to load model with proper error handling
def load_main_model(selected_model):
    try:
        # Construct full model path
        model_path = os.path.join(MODEL_DIR, selected_model)
        
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
def hybrid_generation(query, model_choice, use_external_search, use_chunk_generation):
    # Check if model choice is valid
    if model_choice.startswith("No models found"):
        return "Error: No models available. Please add models to the 'models' directory.", ""
    
    # Load selected model (using cache)
    if model_choice not in main_model_cache:
        model, tokenizer, error = load_main_model(model_choice)
        
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
def create_interface():
    available_models = list_available_models()
    
    with gr.Blocks(title="混合对话生成系统") as interface:
        gr.Markdown("# 混合对话生成系统")
        gr.Markdown("选择主模型、是否启用联网搜索和分块生成策略，系统将整合搜索上下文和动态生成参数生成最终回复，并显示生成统计信息。")
        
        with gr.Row():
            with gr.Column(scale=2):
                query_input = gr.Textbox(lines=2, placeholder="请输入你的问题...")
                model_choice = gr.Dropdown(choices=available_models, label="选择主模型", value=available_models[0] if available_models else None)
                with gr.Row():
                    use_search = gr.Checkbox(label="启用 DuckDuckGo 联网搜索", value=True)
                    use_chunk = gr.Checkbox(label="启用分块生成策略", value=False)
                submit_btn = gr.Button("生成回答")
            
            with gr.Column(scale=1):
                gpu_stats = gr.Textbox(label="实时 GPU 状态", value="等待生成...", interactive=False)
        
        response = gr.Textbox(label="生成回答")
        stats = gr.Textbox(label="生成统计信息")
        
        def update_gpu_stats():
            return gpu_monitor.get_current_stats()
        
        submit_btn.click(
            fn=hybrid_generation,
            inputs=[query_input, model_choice, use_search, use_chunk],
            outputs=[response, stats]
        )
        
        gpu_stats.every(5, update_gpu_stats)  # Update GPU stats every 5 second    
    
    return interface

# ------------------------------
# 6. Main Application
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
    
    # Create and launch interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,       # Match exposed port in Docker
        share=True             # Don't create public URL
    )