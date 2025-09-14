let currentSlide = 0;
let totalSlides = 10;
let formData = {};

// Skin type data and products
const skinTypeData = {
    normal: {
        color: '#f8fafc',
        description: 'Balanced skin with few imperfections',
        recommendations: [
            'Use a gentle cleanser twice daily',
            'Apply a light moisturizer',
            'Use sunscreen daily (SPF 30+)',
            'Maintain your current routine'
        ]
    },
    dry: {
        color: '#fef3c7',
        description: 'Lacks moisture and may feel tight',
        recommendations: [
            'Use a cream-based cleanser',
            'Apply rich moisturizer twice daily',
            'Use a humidifier at night',
            'Avoid hot water when washing'
        ]
    },
    oily: {
        color: '#dcfce7',
        description: 'Produces excess sebum, may have enlarged pores',
        recommendations: [
            'Use a foaming cleanser',
            'Apply oil-free moisturizer',
            'Consider salicylic acid products',
            'Use blotting papers during day'
        ]
    },
    sensitive: {
        color: '#fecaca',
        description: 'Easily irritated, may react to products',
        recommendations: [
            'Use fragrance-free products',
            'Patch test new products',
            'Avoid alcohol-based products',
            'Use lukewarm water for cleansing'
        ]
    }
};

const productRecommendations = {
    normal: [
        {
            name: 'Gentle Daily Cleanser',
            description: 'Mild foam cleanser for balanced skin',
            price: '$24.99',
            image: '🧴'
        },
        {
            name: 'Light Moisturizer',
            description: 'Non-greasy daily moisturizer',
            price: '$32.99',
            image: '🧴'
        },
        {
            name: 'Daily SPF 30',
            description: 'Broad spectrum sun protection',
            price: '$28.99',
            image: '☀️'
        },
        {
            name: 'Vitamin C Serum',
            description: 'Antioxidant protection serum',
            price: '$45.99',
            image: '💧'
        }
    ],
    dry: [
        {
            name: 'Hydrating Cream Cleanser',
            description: 'Gentle cleanser that doesn\'t strip moisture',
            price: '$29.99',
            image: '🧴'
        },
        {
            name: 'Rich Night Cream',
            description: 'Intensive overnight moisturizer',
            price: '$49.99',
            image: '🌙'
        },
        {
            name: 'Hyaluronic Acid Serum',
            description: 'Deep hydration booster',
            price: '$39.99',
            image: '💧'
        },
        {
            name: 'Gentle Exfoliator',
            description: 'Mild exfoliation for dry skin',
            price: '$34.99',
            image: '🧽',
        }
        ]
    };
// --- Skin Type Prediction Form Handler ---
// Only add this handler once, and do not duplicate DOMContentLoaded


const form = document.getElementById('skinAnalysisForm');

if (form) {
    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        const formData = new FormData(form);
        const data = {
            age: parseInt(formData.get('age')),
            gender: formData.get('gender'),
            water_intake_liters: parseFloat(formData.get('water_intake_liters')),
    
            weather: formData.get('weather'),
            oiliness: formData.get('oiliness'),
            acne: formData.get('acne'),
            tightness_after_wash: formData.get('tightness_after_wash'),
            makeup_usage: formData.get('makeup_usage'),
            flaking: formData.get('flaking'),
            redness_itchiness: formData.get('redness_itchiness')
        };
        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            const result = await response.json();
            document.getElementById('result').innerHTML = `<h2>Predicted Skin Type: ${result.skin_type}</h2>`;
        } catch (err) {
            document.getElementById('result').innerHTML = '<span style="color:red">Prediction failed. Is the backend running?</span>';
        }
    });

        
    
    oily: [
        {
            name: 'Foaming Face Wash',
            description: 'Deep cleansing foam for oily skin',
            price: '$26.99',
            image: '🧼'
        },
        {
            name: 'Oil-Free Moisturizer',
            description: 'Lightweight, non-comedogenic formula',
            price: '$31.99',
            image: '🧴'
        },
        {
            name: 'Salicylic Acid Treatment',
            description: 'BHA treatment for pore care',
            price: '$42.99',
            image: '💊'
        },
        {
            name: 'Clay Mask',
            description: 'Weekly deep-cleansing treatment',
            price: '$24.99',
            image: '🎭'
        }]
    
    sensitive: [
        {
            name: 'Gentle Cleanser',
            description: 'Fragrance-free, hypoallergenic',
            price: '$33.99',
            image: '🤲'
        },
        {
            name: 'Soothing Moisturizer',
            description: 'Calming formula for sensitive skin',
            price: '$44.99',
            image: '🌿'
        },
        {
            name: 'Mineral Sunscreen',
            description: 'Physical SPF for sensitive skin',
            price: '$36.99',
            image: '🛡️'
        },
        {
            name: 'Calming Serum',
            description: 'Reduces redness and irritation',
            price: '$52.99',
            image: '🧪'
        }
    ]
};


// DOM Elements
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM Content Loaded - Initializing app...');
    initializeApp();
});

function initializeApp() {
    console.log('Initializing app...');
    setupEventListeners();
    showPage('home');
    updateProgress();
    console.log('App initialized successfully');
}

function setupEventListeners() {
    // Navigation
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    
    if (hamburger) {
        hamburger.addEventListener('click', () => {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });
    }

    // Form submission
    const form = document.getElementById('skinAnalysisForm');
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }

    // Contact form
    const contactForm = document.querySelector('.contact-form');
    if (contactForm) {
        contactForm.addEventListener('submit', handleContactSubmit);
    }

    // Close mobile menu when clicking a link
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', () => {
            if (navMenu && navMenu.classList.contains('active')) {
                navMenu.classList.remove('active');
                hamburger.classList.remove('active');
            }
        });
    });
}

// Navigation Functions
function showPage(pageId) {
    console.log('Showing page:', pageId);
    
    // Hide all pages
    document.querySelectorAll('.page').forEach(page => {
        page.classList.remove('active');
    });
    
    // Show selected page
    const targetPage = document.getElementById(pageId);
    if (targetPage) {
        console.log('Found target page:', pageId);
        targetPage.classList.add('active');
        targetPage.classList.add('fade-in');
    } else {
        console.error('Target page not found:', pageId);
    }
    
    // Update navigation
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    
    const activeLink = document.querySelector(`[onclick="showPage('${pageId}')"]`);
    if (activeLink) {
        activeLink.classList.add('active');
    }

    // Load products if on products page
    if (pageId === 'products') {
        showProducts('normal');
    }
}

// Questionnaire Functions
function changeSlide(direction) {
    const slides = document.querySelectorAll('.question-slide');
    const currentSlideElement = slides[currentSlide];
    
    // Validate current slide before proceeding
    if (direction > 0 && !validateCurrentSlide()) {
        showValidationMessage();
        return;
    }
    
    // Remove active class from current slide
    currentSlideElement.classList.remove('active');
    
    // Update slide index
    currentSlide += direction;
    
    // Ensure slide index is within bounds
    if (currentSlide < 0) currentSlide = 0;
    if (currentSlide >= totalSlides) currentSlide = totalSlides - 1;
    
    // Show new slide
    slides[currentSlide].classList.add('active');
    slides[currentSlide].classList.add('slide-in-right');
    
    // Update progress and navigation
    updateProgress();
    updateNavigation();
    
    // Update question counter
    document.getElementById('currentQuestion').textContent = currentSlide + 1;
}

function validateCurrentSlide() {
    const currentSlideElement = document.querySelectorAll('.question-slide')[currentSlide];
    const inputs = currentSlideElement.querySelectorAll('input[required]');
    
    for (let input of inputs) {
        if (input.type === 'radio') {
            const radioGroup = currentSlideElement.querySelectorAll(`input[name="${input.name}"]`);
            const isChecked = Array.from(radioGroup).some(radio => radio.checked);
            if (!isChecked) return false;
        } else if (input.type === 'number') {
            if (!input.value || input.value === '') return false;
            const value = parseFloat(input.value);
            const min = parseFloat(input.min);
            const max = parseFloat(input.max);
            if (value < min || value > max) return false;
        }
    }
    return true;
}

function showValidationMessage() {
    // Create and show validation message
    const existingMessage = document.querySelector('.validation-message');
    if (existingMessage) {
        existingMessage.remove();
    }
    
    const message = document.createElement('div');
    message.className = 'validation-message';
    message.style.cssText = `
        background: #fecaca;
        color: #b91c1c;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
        text-align: center;
        font-weight: 600;
        animation: fadeIn 0.3s ease;
    `;
    message.textContent = 'Please answer this question before continuing.';
    
    const currentSlideElement = document.querySelectorAll('.question-slide')[currentSlide];
    currentSlideElement.appendChild(message);
    
    setTimeout(() => {
        if (message.parentNode) {
            message.remove();
        }
    }, 3000);
}

function updateProgress() {
    const progressBar = document.getElementById('progress');
    const progressPercentage = ((currentSlide + 1) / totalSlides) * 100;
    progressBar.style.width = `${progressPercentage}%`;
}

function updateNavigation() {
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');
    const submitBtn = document.getElementById('submitBtn');
    
    // Update previous button
    prevBtn.disabled = currentSlide === 0;
    
    // Update next/submit buttons
    if (currentSlide === totalSlides - 1) {
        nextBtn.style.display = 'none';
        submitBtn.style.display = 'inline-block';
    } else {
        nextBtn.style.display = 'inline-block';
        submitBtn.style.display = 'none';
    }
}

function resetAnalysis() {
    currentSlide = 0;
    formData = {};
    
    // Reset form
    const form = document.getElementById('skinAnalysisForm');
    if (form) {
        form.reset();
    }
    
    // Show first slide
    document.querySelectorAll('.question-slide').forEach((slide, index) => {
        slide.classList.remove('active');
        if (index === 0) slide.classList.add('active');
    });
    
    // Hide results and show form
    document.getElementById('results').style.display = 'none';
    document.getElementById('questionForm').style.display = 'block';
    
    // Update navigation
    updateProgress();
    updateNavigation();
    
    // Update question counter
    document.getElementById('currentQuestion').textContent = '1';
}

// Form Handling
async function handleFormSubmit(e) {
    e.preventDefault();
    console.log('Form submitted!');
    
    if (!validateCurrentSlide()) {
        showValidationMessage();
        return;
    }
    
    // Collect form data more comprehensively
    const form = e.target;
    formData = {};
    
    // Get all form inputs
    const inputs = form.querySelectorAll('input, select, textarea');
    
    inputs.forEach(input => {
        if (input.name) {
            if (input.type === 'radio') {
                if (input.checked) {
                    formData[input.name] = input.value;
                }
            } else if (input.type === 'checkbox') {
                formData[input.name] = input.checked;
            } else if (input.type === 'number') {
                formData[input.name] = parseFloat(input.value) || 0;
            } else {
                formData[input.name] = input.value;
            }
        }
    });
    
    console.log('Form data collected:', formData);
    
    // Validate that we have all required fields
    const requiredFields = ['age', 'gender', 'water_intake', 'weather', 'oiliness', 
                           'acne', 'tightness_after_wash', 'makeup_usage', 'flaking', 'redness_itchiness'];
    
    const missingFields = requiredFields.filter(field => !formData[field]);
    if (missingFields.length > 0) {
        console.error('Missing required fields:', missingFields);
        showNotification(`Please fill in all required fields: ${missingFields.join(', ')}`, 'error');
        return;
    }
    
    // Show loading state
    showLoadingState();
    
    try {
        // Get prediction (either from API or local algorithm)
        const prediction = await predictSkinType(formData);
        console.log('Prediction result:', prediction);
        displayResults(prediction);
    } catch (error) {
        console.error('Prediction error:', error);
        showNotification('Error making prediction. Please try again.', 'error');
        
        // Reset button state
        const submitBtn = document.getElementById('submitBtn');
        if (submitBtn) {
            submitBtn.innerHTML = 'Get Results';
            submitBtn.disabled = false;
        }
    }
}

function showLoadingState() {
    const submitBtn = document.getElementById('submitBtn');
    const originalText = submitBtn.innerHTML;
    
    submitBtn.innerHTML = '<span class="loading"></span> Analyzing...';
    submitBtn.disabled = true;
    
    // Reset button after processing
    setTimeout(() => {
        submitBtn.innerHTML = originalText;
        submitBtn.disabled = false;
    }, 2000);
}

// Machine Learning Prediction
async function predictSkinType(data) {
    try {
        // Try to use the Python API first
    const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        if (response.ok) {
            const result = await response.json();
            return {
                skinType: result.skin_type,
                confidence: result.confidence,
                allScores: result.all_scores,
                recommendations: result.recommendations
            };
        } else {
            // If API fails, fall back to local prediction
            console.warn('API prediction failed, using local algorithm');
            return predictSkinTypeLocal(data);
        }
    } catch (error) {
        // If API is not available, use local prediction
        console.warn('API not available, using local algorithm');
        return predictSkinTypeLocal(data);
    }
}

// Local prediction algorithm as fallback
function predictSkinTypeLocal(data) {
    let scores = {
        normal: 25,
        dry: 25,
        oily: 25,
        sensitive: 25
    };
    
    // Age factor
    if (data.age > 40) {
        scores.dry += 10;
        scores.sensitive += 5;
    }
    
    // Gender factor
    if (data.gender === 'female') {
        scores.sensitive += 5;
    }
    
    // Water intake
    if (data.water_intake < 2) {
        scores.dry += 15;
    }
    
    // Weather
    if (data.weather === 'dry' || data.weather === 'hot') {
        scores.dry += 10;
        scores.oily += 5;
    } else if (data.weather === 'cold') {
        scores.dry += 10;
        scores.sensitive += 5;
    } else if (data.weather === 'humid') {
        scores.oily += 10;
    }
    
    // Oiliness level
    if (data.oiliness === 'high') {
        scores.oily += 20;
        scores.dry -= 10;
    } else if (data.oiliness === 'low') {
        scores.dry += 15;
        scores.oily -= 10;
    }
    
    // Acne
    if (data.acne === 'yes') {
        scores.oily += 15;
        scores.sensitive += 5;
    }
    
    // Tightness after wash
    if (data.tightness_after_wash === 'yes') {
        scores.dry += 15;
        scores.sensitive += 5;
    }
    
    // Makeup usage
    if (data.makeup_usage === 'frequent') {
        scores.sensitive += 5;
        scores.oily += 5;
    }
    
    // Flaking
    if (data.flaking === 'yes') {
        scores.dry += 20;
        scores.sensitive += 10;
    }
    
    // Redness/itchiness
    if (data.redness_itchiness === 'yes') {
        scores.sensitive += 20;
    }
    
    // Normalize scores
    const total = Object.values(scores).reduce((sum, score) => sum + score, 0);
    const probabilities = {};
    
    for (let skinType in scores) {
        probabilities[skinType] = Math.max(0, scores[skinType]) / total;
    }
    
    // Find the highest scoring skin type
    const predictedType = Object.keys(probabilities).reduce((a, b) => 
        probabilities[a] > probabilities[b] ? a : b
    );
    
    return {
        skinType: predictedType,
        confidence: probabilities[predictedType],
        allScores: probabilities
    };
}

function displayResults(prediction) {
    // Hide form and show results
    document.getElementById('questionForm').style.display = 'none';
    document.getElementById('results').style.display = 'block';
    
    const skinType = prediction.skinType;
    const skinTypeInfo = skinTypeData[skinType];
    
    // Update result display
    const resultIcon = document.getElementById('resultIcon');
    resultIcon.className = `skin-type-icon ${skinType}`;
    
    const skinTypeResult = document.getElementById('skinTypeResult');
    skinTypeResult.textContent = skinType.charAt(0).toUpperCase() + skinType.slice(1);
    skinTypeResult.style.color = getColorForSkinType(skinType);
    
    const confidenceScore = document.getElementById('confidenceScore');
    confidenceScore.textContent = `${(prediction.confidence * 100).toFixed(1)}% confidence`;
    
    // Display confidence breakdown
    displayConfidenceBreakdown(prediction.allScores);
    
    // Display recommendations (use API recommendations if available, otherwise local)
    const recommendations = prediction.recommendations || skinTypeInfo.recommendations;
    displayRecommendations(skinType, recommendations);
}

function getColorForSkinType(skinType) {
    const colors = {
        normal: '#6b7280',
        dry: '#d97706',
        oily: '#059669',
        sensitive: '#dc2626'
    };
    return colors[skinType] || '#6b7280';
}

function displayConfidenceBreakdown(scores) {
    const container = document.getElementById('confidenceBreakdown');
    container.innerHTML = '<h3>Confidence Breakdown</h3>';
    
    Object.entries(scores).forEach(([skinType, score]) => {
        const percentage = (score * 100).toFixed(1);
        
        const item = document.createElement('div');
        item.className = 'confidence-item';
        item.innerHTML = `
            <div class="confidence-label">${skinType}</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: 0%"></div>
            </div>
            <div class="confidence-percentage">${percentage}%</div>
        `;
        
        container.appendChild(item);
        
        // Animate the bar
        setTimeout(() => {
            const fill = item.querySelector('.confidence-fill');
            fill.style.width = `${percentage}%`;
        }, 100);
    });
}

function displayRecommendations(skinType, customRecommendations = null) {
    const container = document.getElementById('recommendations');
    const recommendations = customRecommendations || skinTypeData[skinType].recommendations;
    
    container.innerHTML = `
        <h3>Skincare Recommendations for ${skinType.charAt(0).toUpperCase() + skinType.slice(1)} Skin</h3>
        <ul class="recommendation-list">
            ${recommendations.map(rec => `
                <li><i class="fas fa-check"></i> ${rec}</li>
            `).join('')}
        </ul>
    `;
}

// Products Page
function showProducts(skinType) {
    // Update active tab
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    event.target.classList.add('active');
    
    const products = productRecommendations[skinType];
    const container = document.getElementById('productsGrid');
    
    container.innerHTML = products.map(product => `
        <div class="product-card fade-in">
            <div class="product-image">${product.image}</div>
            <div class="product-info">
                <h3>${product.name}</h3>
                <p>${product.description}</p>
                <div class="product-price">${product.price}</div>
                <button class="product-btn" onclick="addToCart('${product.name}')">
                    Add to Cart
                </button>
            </div>
        </div>
    `).join('');
}

function addToCart(productName) {
    // Simple cart functionality
    showNotification(`${productName} added to cart!`, 'success');
}

function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: ${type === 'success' ? '#10b981' : '#667eea'};
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        z-index: 10000;
        animation: slideInRight 0.3s ease;
        font-weight: 600;
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'fadeOut 0.3s ease';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 300);
    }, 3000);
}

// FAQ Functions
function toggleFAQ(element) {
    const faqItem = element.parentElement;
    const isActive = faqItem.classList.contains('active');
    
    // Close all FAQ items
    document.querySelectorAll('.faq-item').forEach(item => {
        item.classList.remove('active');
    });
    
    // Open clicked item if it wasn't active
    if (!isActive) {
        faqItem.classList.add('active');
    }
}

// Contact Form
function handleContactSubmit(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData);
    
    // Show loading state
    const submitBtn = e.target.querySelector('.submit-btn');
    const originalText = submitBtn.textContent;
    submitBtn.innerHTML = '<span class="loading"></span> Sending...';
    submitBtn.disabled = true;
    
    // Simulate form submission
    setTimeout(() => {
        showNotification('Message sent successfully! We\'ll get back to you soon.', 'success');
        e.target.reset();
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
    }, 2000);
}

// Utility Functions
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Initialize products page when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Set up default product tab if on products page
    const productsPage = document.getElementById('products');
    if (productsPage && productsPage.classList.contains('active')) {
        showProducts('normal');
    }
});

// Add smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add keyboard navigation for accessibility
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        // Close mobile menu if open
        const navMenu = document.querySelector('.nav-menu');
        const hamburger = document.querySelector('.hamburger');
        
        if (navMenu && navMenu.classList.contains('active')) {
            navMenu.classList.remove('active');
            hamburger.classList.remove('active');
        }
        
        // Close any open FAQ items
        document.querySelectorAll('.faq-item.active').forEach(item => {
            item.classList.remove('active');
        });
    }
});
