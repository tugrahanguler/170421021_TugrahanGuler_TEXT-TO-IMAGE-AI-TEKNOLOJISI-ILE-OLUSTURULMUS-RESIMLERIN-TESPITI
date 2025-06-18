const mobileMenuBtn = document.querySelector('.mobile-menu-btn');
const navLinks = document.querySelector('.nav-links');

mobileMenuBtn.addEventListener('click', () => {
    navLinks.classList.toggle('active');
});

const API_URL = 'http://localhost:5000/api';

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const imagePreview = document.getElementById('imagePreview');
const fileName = document.getElementById('fileName');
const removeFile = document.getElementById('removeFile');
const analyzeBtn = document.getElementById('analyzeBtn');

let lastDroppedFile = null;

uploadArea.querySelector('.browse-btn').addEventListener('click', (e) => {
    e.preventDefault();
    fileInput.click();
});

['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, preventDefaults, false);
}); 

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    uploadArea.addEventListener(eventName, highlight, false);
});

['dragleave', 'drop'].forEach(eventName => {
    uploadArea.addEventListener(eventName, unhighlight, false);
});

function highlight() {
    uploadArea.classList.add('drag-over');
}

function unhighlight() {
    uploadArea.classList.remove('drag-over');
}

uploadArea.addEventListener('drop', handleDrop, false);

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length) {
        lastDroppedFile = files[0];
        handleFiles(files);
    }
}

fileInput.addEventListener('change', () => {
    if (fileInput.files.length) {
        lastDroppedFile = null; 
        handleFiles(fileInput.files);
    }
});

function handleFiles(files) {
    const file = files[0];
    
    try {
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInput.files = dataTransfer.files;
    } catch (error) {
        console.error('DataTransfer API hatası:', error);
    }
    
    if (!file.type.match('image.*')) {
        alert('Lütfen bir görsel dosyası yükleyin!');
        return;
    }

    const reader = new FileReader();
    
    reader.onload = function(e) {
        imagePreview.src = e.target.result;
        fileName.textContent = file.name;
        previewContainer.classList.add('active');
        analyzeBtn.disabled = false;
        
        previewContainer.style.animation = 'none';
        setTimeout(() => {
            previewContainer.style.animation = 'fadeIn 0.5s ease';
        }, 10);
    };
    
    reader.readAsDataURL(file);
}

removeFile.addEventListener('click', () => {
    previewContainer.classList.remove('active');
    fileInput.value = '';
    lastDroppedFile = null;
    analyzeBtn.disabled = true;
    
    const resultContainer = document.getElementById('resultContainer');
    if (resultContainer) {
        resultContainer.remove();
    }
    
    setTimeout(() => {
        imagePreview.src = '';
    }, 300);
});

analyzeBtn.addEventListener('click', async () => {
    let fileToAnalyze = null;
    
    if (fileInput.files.length) {
        fileToAnalyze = fileInput.files[0];
    } else if (lastDroppedFile) {
        fileToAnalyze = lastDroppedFile;
    } else if (imagePreview.src && imagePreview.src.startsWith('data:image')) {
        try {
            const response = await fetch(imagePreview.src);
            fileToAnalyze = await response.blob();
        } catch (error) {
            console.error('Görüntü dönüştürme hatası:', error);
        }
    }
    
    if (!fileToAnalyze) {
        alert('Lütfen önce bir görsel yükleyin!');
        return;
    }
    
    analyzeBtn.classList.add('loading');
    
    try {
        const formData = new FormData();
        formData.append('image', fileToAnalyze);
        
        const response = await fetch(`${API_URL}/analyze`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const result = await response.json();
        
        displayResult(result);
        
    } catch (error) {
        console.error('Analiz sırasında hata oluştu:', error);
        alert(`Analiz sırasında bir hata oluştu: ${error.message}`);
    } finally {
        analyzeBtn.classList.remove('loading');
    }
});

function displayResult(result) {
    const existingResult = document.getElementById('resultContainer');
    if (existingResult) {
        existingResult.remove();
    }
    
    const resultContainer = document.createElement('div');
    resultContainer.id = 'resultContainer';
    resultContainer.className = 'result-container';
    
    const isReal = result.prediction === 'real';
    const resultColor = isReal ? 'var(--success)' : 'var(--danger)';
    const resultText = isReal ? 'Gerçek Görüntü' : 'Yapay Zeka Üretimi';
    
    resultContainer.innerHTML = `
        <div class="result-header" style="background-color: ${resultColor}">
            <h3>Analiz Sonucu</h3>
        </div>
        <div class="result-content">
            <div class="result-main">
                <div class="result-icon" style="color: ${resultColor}">
                    <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        ${isReal ? 
                            '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline>' : 
                            '<circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line>'}
                    </svg>
                </div>
                <div class="result-details">
                    <h4>${resultText}</h4>
                    <p>Güven Oranı: <strong>%${result.confidence.toFixed(2)}</strong></p>
                </div>
            </div>
            
            <div class="probability-bars">
                <div class="probability-item">
                    <div class="probability-label">
                        <span>Gerçek</span>
                        <span>${result.class_probabilities.real.toFixed(2)}%</span>
                    </div>
                    <div class="probability-bar">
                        <div class="probability-fill" style="width: ${result.class_probabilities.real}%; background-color: var(--success)"></div>
                    </div>
                </div>
                <div class="probability-item">
                    <div class="probability-label">
                        <span>Yapay</span>
                        <span>${result.class_probabilities.fake.toFixed(2)}%</span>
                    </div>
                    <div class="probability-bar">
                        <div class="probability-fill" style="width: ${result.class_probabilities.fake}%; background-color: var(--danger)"></div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    analyzeBtn.parentNode.insertBefore(resultContainer, analyzeBtn.nextSibling);
    
    setTimeout(() => {
        resultContainer.classList.add('active');
    }, 10);
}

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, { threshold: 0.1 });

document.querySelectorAll('.feature-card').forEach(card => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(20px)';
    card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(card);
});

async function checkBackendStatus() {
    try {
        const response = await fetch(`${API_URL}/status`);
        if (response.ok) {
            const data = await response.json();
            console.log('Backend durumu:', data);
            
            if (!data.model_loaded) {
                console.warn('Model yüklenemedi. Analiz fonksiyonu çalışmayabilir.');
            }
        }
    } catch (error) {
        console.error('Backend bağlantısı kurulamadı:', error);
    }
}

window.addEventListener('load', checkBackendStatus);
