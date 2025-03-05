// 脚本用于处理和拦截404资源请求

document.addEventListener('DOMContentLoaded', function() {
  console.log("Running 404 error prevention script");
  
  // 创建一个网络请求拦截器
  if (window.fetch) {
    const originalFetch = window.fetch;
    window.fetch = function(resource, options) {
      // 检查请求URL是否包含常见的404资源
      if (typeof resource === 'string') {
        if (
          resource.includes('.woff2') || 
          resource.includes('system-ui') || 
          resource.includes('ui-sans-serif')
        ) {
          console.log(`Prevented fetch for: ${resource}`);
          // 返回空响应而不是404
          return Promise.resolve(new Response('', {
            status: 200,
            headers: {'Content-Type': 'application/octet-stream'}
          }));
        }
        
        // 如果是manifest.json但不是完整URL（相对路径）
        if (resource === 'manifest.json' || resource.endsWith('/manifest.json')) {
          console.log(`Redirecting manifest.json request to our version`);
          // 重定向到我们的manifest.json
          return originalFetch('file=manifest.json', options);
        }
      }
      
      // 对于所有其他请求，使用原始fetch
      return originalFetch.apply(this, arguments);
    };
  }
  
  console.log("404 error prevention script loaded");
});

// 为各种系统字体创建空的preload指令，防止浏览器尝试加载它们
function createPreloadLink(href, type) {
  const link = document.createElement('link');
  link.rel = 'preload';
  link.as = 'font';
  link.href = 'data:font/woff2;base64,';  // 空的base64编码字体
  link.type = type || 'font/woff2';
  link.crossOrigin = 'anonymous';
  document.head.appendChild(link);
}

// 添加常见的系统字体preload
['system-ui-Regular.woff2', 'system-ui-Bold.woff2', 
 'ui-sans-serif-Regular.woff2', 'ui-sans-serif-Bold.woff2'].forEach(font => {
  createPreloadLink(font);
});
