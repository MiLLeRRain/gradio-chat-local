// Custom JavaScript for enhancing UI interactivity

document.addEventListener('DOMContentLoaded', () => {
    // Function to apply styles to elements once they're loaded
    function enhanceUI() {
        // Add animation effect to the header
        const header = document.querySelector('.app-header h1');
        if (header) {
            header.style.animation = 'fadeIn 1s ease-in-out';
        }
        
        // Make chatbot container resizable
        const chatbotContainer = document.querySelector('.gradio-chatbot');
        if (chatbotContainer) {
            // Enable vertical resizing
            chatbotContainer.style.resize = 'vertical';
            chatbotContainer.style.overflow = 'auto';
            chatbotContainer.style.minHeight = '400px';
            chatbotContainer.style.maxHeight = '80vh';
            chatbotContainer.style.transition = 'height 0.3s';
            
            // Add a visual resize handle
            const handle = document.createElement('div');
            handle.className = 'resize-handle';
            handle.style.cssText = `
                height: 8px;
                background-color: rgba(98, 0, 238, 0.3);
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                cursor: ns-resize;
                transition: background-color 0.3s;
                border-radius: 0 0 8px 8px;
                z-index: 10;
            `;
            
            // Add hover effect
            handle.addEventListener('mouseover', () => {
                handle.style.backgroundColor = 'rgba(98, 0, 238, 0.8)';
            });
            
            handle.addEventListener('mouseout', () => {
                handle.style.backgroundColor = 'rgba(98, 0, 238, 0.3)';
            });
            
            // Parent container needs position relative
            chatbotContainer.parentElement.style.position = 'relative';
            chatbotContainer.parentElement.appendChild(handle);
        }
        
        // Add subtle animation to buttons on hover
        const buttons = document.querySelectorAll('button');
        buttons.forEach(button => {
            button.addEventListener('mouseover', () => {
                button.style.transform = 'translateY(-2px)';
                button.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.2)';
            });
            
            button.addEventListener('mouseout', () => {
                button.style.transform = 'translateY(0)';
                button.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.1)';
            });
        });
    }
    
    // Try to enhance UI immediately
    enhanceUI();
    
    // Also set a timeout to catch elements that load later
    setTimeout(enhanceUI, 1000);
    setTimeout(enhanceUI, 2000);
});

// Add keypress shortcuts
document.addEventListener('keydown', (event) => {
    // Ctrl+Enter to submit
    if (event.ctrlKey && event.key === 'Enter') {
        const submitButton = document.querySelector('.submit-button button');
        if (submitButton) {
            submitButton.click();
        }
    }
});

// Add custom animations
document.head.insertAdjacentHTML('beforeend', `
    <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(98, 0, 238, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(98, 0, 238, 0); }
            100% { box-shadow: 0 0 0 0 rgba(98, 0, 238, 0); }
        }
        
        .submit-button button {
            animation: pulse 2s infinite;
        }
        
        .chatbot-message-appear {
            animation: fadeIn 0.5s ease-out forwards;
        }
    </style>
`);
