// Crystal background animation with Three.js

// Create the Three.js background
function initThreeJSBackground() {
    // Create canvas element
    const canvas = document.createElement('canvas');
    canvas.id = 'background';
    canvas.style.position = 'fixed';
    canvas.style.top = '0';
    canvas.style.left = '0';
    canvas.style.width = '100%';
    canvas.style.height = '100%';
    canvas.style.zIndex = '-1';
    canvas.style.opacity = '0.7';
    document.body.prepend(canvas);
    
    // Set up Three.js scene
    const scene = new THREE.Scene();
    const clock = new THREE.Clock();
    
    // Use a gradient background that matches our theme
    const bgColor = new THREE.Color(0xf5f8ff);
    scene.background = bgColor;
    
    // Camera setup
    const camera = new THREE.PerspectiveCamera(70, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 15);
    camera.lookAt(0, 0, 0);
    
    // Renderer setup
    const renderer = new THREE.WebGLRenderer({
        canvas: document.getElementById('background'),
        antialias: true,
        alpha: true
    });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(window.innerWidth, window.innerHeight);
    
    // Add lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    const spotLight1 = new THREE.SpotLight(0x6200ee, 1.5);
    spotLight1.position.set(200, 100, 50);
    const spotLight2 = new THREE.SpotLight(0x03dac5, 1.5);
    spotLight2.position.set(-200, -100, -50);
    const spotLight3 = new THREE.SpotLight(0xffffff, 0.8);
    spotLight3.position.set(0, 50, 200);
    
    scene.add(ambientLight, spotLight1, spotLight2, spotLight3);
    
    // Crystal color palettes for our theme
    const primaryColors = {
        start: new THREE.Color(0x6200ee),
        end: new THREE.Color(0xa675ff)
    };
    
    const secondaryColors = {
        start: new THREE.Color(0x03dac5),
        end: new THREE.Color(0x66fff8)
    };
    
    const accentColors = {
        start: new THREE.Color(0xff79b0),
        end: new THREE.Color(0xffc9e3)
    };
    
    // Crystal objects container
    const crystals = [];
    
    // Crystal factory function
    function createCrystal(type) {
        let geometry;
        let colorPair;
        let scale = THREE.MathUtils.randFloat(0.5, 1.5);
        
        switch(type) {
            case 'diamond':
                geometry = new THREE.OctahedronGeometry(1, 0);
                colorPair = primaryColors;
                break;
            case 'prism':
                geometry = new THREE.DodecahedronGeometry(1, 0);
                colorPair = secondaryColors;
                break;
            case 'gem':
                geometry = new THREE.IcosahedronGeometry(1, 0);
                colorPair = accentColors;
                break;
            default:
                geometry = new THREE.TetrahedronGeometry(1, 0);
                colorPair = primaryColors;
        }
        
        // Create shiny, crystalline material
        const material = new THREE.MeshPhysicalMaterial({
            color: colorPair.start,
            metalness: 0.2,
            roughness: 0.1,
            reflectivity: 1,
            clearcoat: 1.0,
            clearcoatRoughness: 0.1,
            transparent: true,
            opacity: 0.8,
            side: THREE.DoubleSide
        });
        
        const crystal = new THREE.Mesh(geometry, material);
        
        // Randomize position within bounds
        const posRange = 25;
        crystal.position.set(
            THREE.MathUtils.randFloatSpread(posRange),
            THREE.MathUtils.randFloatSpread(posRange),
            THREE.MathUtils.randFloatSpread(posRange / 2) - posRange / 4
        );
        
        // Randomize rotation
        crystal.rotation.set(
            Math.random() * Math.PI,
            Math.random() * Math.PI,
            Math.random() * Math.PI
        );
        
        // Randomize scale (keeping the original proportions)
        crystal.scale.set(scale, scale, scale);
        
        // Store the original color for animation
        crystal.userData = {
            colorStart: colorPair.start.clone(),
            colorEnd: colorPair.end.clone(),
            rotationSpeed: {
                x: THREE.MathUtils.randFloatSpread(0.001) * 0.5,
                y: THREE.MathUtils.randFloatSpread(0.001) * 0.5,
                z: THREE.MathUtils.randFloatSpread(0.001) * 0.5
            },
            initialPosition: crystal.position.clone(),
            floatOffset: Math.random() * Math.PI * 2,
            floatSpeed: THREE.MathUtils.randFloat(0.2, 0.6),
            floatAmplitude: THREE.MathUtils.randFloat(0.1, 0.3)
        };
        
        scene.add(crystal);
        return crystal;
    }
    
    // Create crystals of different types
    const crystalCount = {
        diamond: 15,
        prism: 15,
        gem: 15
    };
    
    Object.entries(crystalCount).forEach(([type, count]) => {
        for (let i = 0; i < count; i++) {
            crystals.push(createCrystal(type));
        }
    });
    
    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        
        const time = clock.getElapsedTime();
        
        crystals.forEach(crystal => {
            // Smooth rotation
            crystal.rotation.x += crystal.userData.rotationSpeed.x;
            crystal.rotation.y += crystal.userData.rotationSpeed.y;
            crystal.rotation.z += crystal.userData.rotationSpeed.z;
            
            // Gentle floating motion
            const floatY = Math.sin(time * crystal.userData.floatSpeed + crystal.userData.floatOffset) * crystal.userData.floatAmplitude;
            crystal.position.y = crystal.userData.initialPosition.y + floatY;
            
            // Color pulsing effect
            const pulseIntensity = (Math.sin(time * 0.5 + crystal.userData.floatOffset) + 1) * 0.5;
            crystal.material.color.copy(crystal.userData.colorStart).lerp(crystal.userData.colorEnd, pulseIntensity);
            
            // Subtle opacity pulsing
            crystal.material.opacity = 0.7 + Math.sin(time * 0.3 + crystal.userData.floatOffset) * 0.15;
        });
        
        // Camera slight motion to add depth perception
        camera.position.x = Math.sin(time * 0.1) * 2;
        camera.position.y = Math.cos(time * 0.1) * 2;
        camera.lookAt(0, 0, 0);
        
        renderer.render(scene, camera);
    }
    
    // Handle window resize
    function onWindowResize() {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    }
    
    window.addEventListener('resize', onWindowResize);
    
    // Start animation
    animate();
}

// Initialize when document is fully loaded
document.addEventListener('DOMContentLoaded', () => {
    // Check if Three.js is loaded
    if (typeof THREE === 'undefined') {
        console.error('Three.js not loaded! Make sure it is included before this script.');
        return;
    }
    
    // Initialize background
    console.log('Initializing Three.js background');
    try {
        initThreeJSBackground();
    } catch (error) {
        console.error('Error initializing Three.js background:', error);
    }
});
