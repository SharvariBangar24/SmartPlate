// frontend/script.js

// DOM Elements
const scanBtnHero = document.getElementById('scan-btn');
const scannerSection = document.getElementById('scanner');
const uploadImageInput = document.getElementById('upload-image');
const takePhotoInput = document.getElementById('take-photo');
const imagePreview = document.getElementById('image-preview');
const placeholderText = document.getElementById('placeholder-text');
const analyzeBtn = document.getElementById('analyze-btn');

const loadingSection = document.getElementById('loading-section');
const resultsSection = document.getElementById('results');
const resetBtn = document.getElementById('reset-btn');

// Dummy Data for Analysis (simulating a backend/AI response)
const dummyAnalysisData = {
    foods: [
        { name: "Grilled Salmon", confidence: "96%" },
        { name: "Quinoa", confidence: "91%" },
        { name: "Roasted Asparagus", confidence: "98%" }
    ],
    nutrition: {
        calories: 420,
        protein: "38g",
        carbs: "25g",
        fat: "18g"
    },
    insight: "This is a fantastic meal! Salmon provides great omega-3 fatty acids for brain health, while quinoa offers a complete protein source and complex carbohydrates. The asparagus is packed with fiber and vitamins."
};

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    
    // Smooth scroll to scanner section when hero button is clicked
    scanBtnHero.addEventListener('click', () => {
        scannerSection.scrollIntoView({ behavior: 'smooth' });
    });

    // Listen for file selection on "Upload Image"
    uploadImageInput.addEventListener('change', handleImageSelection);
    
    // Listen for file selection on "Take Photo"
    takePhotoInput.addEventListener('change', handleImageSelection);

    // Perform analysis when button is clicked
    analyzeBtn.addEventListener('click', performAnalysis);

    // Reset app for a new scan
    resetBtn.addEventListener('click', resetApp);
});

/**
 * Reads the selected image file and displays it in the preview area
 */
function handleImageSelection(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            // Display the image
            imagePreview.src = e.target.result;
            imagePreview.hidden = false;
            
            // Hide the placeholder text
            placeholderText.hidden = true;
            
            // Show the analyze button
            analyzeBtn.hidden = false;
        }
        
        // Read the image file as a Data URL (base64 string)
        reader.readAsDataURL(file);
    }
}

/**
 * Simulates a delay to mimic an AI API analyzing the image
 */
function performAnalysis() {
    // Hide scanner, show loading spinner
    scannerSection.hidden = true;
    loadingSection.hidden = false;
    
    // Scroll to the loading area smoothly
    loadingSection.scrollIntoView({ behavior: 'smooth' });

    // Simulate a network/AI delay of 2.5 seconds
    setTimeout(() => {
        populateResults(dummyAnalysisData);
        loadingSection.hidden = true;
        resultsSection.hidden = false;
    }, 2500);
}

/**
 * Populates the HTML dashboard with the provided data
 */
function populateResults(data) {
    // Populate Detected Foods List
    const foodList = document.getElementById('food-list');
    foodList.innerHTML = ''; // Clear previous items
    
    data.foods.forEach(food => {
        const li = document.createElement('li');
        li.innerHTML = `<span>${food.name}</span> <span class="food-confidence">${food.confidence}</span>`;
        foodList.appendChild(li);
    });

    // Populate Macros
    document.getElementById('cal-val').textContent = data.nutrition.calories;
    document.getElementById('pro-val').textContent = data.nutrition.protein;
    document.getElementById('carb-val').textContent = data.nutrition.carbs;
    document.getElementById('fat-val').textContent = data.nutrition.fat;

    // Populate Health Insight
    document.getElementById('health-insight-text').textContent = data.insight;
}

/**
 * Resets the application state back to the beginning
 */
function resetApp() {
    // Clear file inputs so the same file can be selected again if needed
    uploadImageInput.value = '';
    takePhotoInput.value = '';
    
    // Reset preview area
    imagePreview.src = '';
    imagePreview.hidden = true;
    placeholderText.hidden = false;
    analyzeBtn.hidden = true;
    
    // Hide results, show scanner again
    resultsSection.hidden = true;
    scannerSection.hidden = false;
    
    // Scroll back up to the scanner section
    scannerSection.scrollIntoView({ behavior: 'smooth' });
}
