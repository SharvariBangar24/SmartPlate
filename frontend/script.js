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

        reader.onload = function (e) {
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
async function performAnalysis() {
    // Hide scanner, show loading spinner
    scannerSection.hidden = true;
    loadingSection.hidden = false;

    // Scroll to loading area
    loadingSection.scrollIntoView({ behavior: 'smooth' });

    let file = uploadImageInput.files[0] || takePhotoInput.files[0];
    if (!file) {
        alert("Please select an image first.");
        loadingSection.hidden = true;
        scannerSection.hidden = false;
        return;
    }

    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('http://127.0.0.1:5000/api/analyze', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        console.log("Backend response:", data);

        if (!response.ok) {
            throw new Error(data.error || 'Unknown error occurred.');
        }

        let insightMsg = `Detected ${data.detection_count} food items.`;
        if (data.detection_count === 0) {
            insightMsg = "No food items detected. Try another image.";
        }
        
        let mappedFoods = (data.foods || []).map(f => {
            let portionStr = '';
            if (f.nutrition && typeof f.nutrition.estimated_weight_g === 'number') {
                portionStr = `~${Math.round(f.nutrition.estimated_weight_g)}g`;
            }
            return {
                name: f.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
                confidence: (f.confidence * 100).toFixed(1) + '%',
                portion: portionStr
            };
        });

        // Use actual nutrition data if available, otherwise 0
        const nutri = data.total_nutrition || {calories: 0, protein: 0, carbs: 0, fat: 0, estimated_weight_g: 0};
        const totalWeightStr = (typeof nutri.estimated_weight_g === 'number' && nutri.estimated_weight_g > 0)
            ? `${Math.round(nutri.estimated_weight_g)}g`
            : '0g';

        const testData = {
            foods: mappedFoods,
            nutrition: {
                calories: Math.round(nutri.calories || 0),
                weight: totalWeightStr,
                protein: (nutri.protein || 0).toFixed(1) + 'g',
                carbs: (nutri.carbs || 0).toFixed(1) + 'g',
                fat: (nutri.fat || 0).toFixed(1) + 'g'
            },
            insight: insightMsg + (data.disclaimer ? '\n\nNote: ' + data.disclaimer : '')
        };

        populateResults(testData);

        loadingSection.hidden = true;
        resultsSection.hidden = false;

    } catch (error) {
        console.error("Backend connection error:", error);

        loadingSection.hidden = true;
        scannerSection.hidden = false;

        alert("Error analyzing image: " + error.message);
    }
}
/**
 * Populates the HTML dashboard with the provided data
 */
function populateResults(data) {
    // Populate Detected Foods List
    const foodList = document.getElementById('food-list');
    foodList.innerHTML = ''; // Clear previous items

    if (!data.foods || data.foods.length === 0) {
        const emptyLi = document.createElement('li');
        emptyLi.className = 'empty-state';
        emptyLi.textContent = 'No food items detected on this plate.';
        foodList.appendChild(emptyLi);
    } else {
        data.foods.forEach(food => {
            const li = document.createElement('li');
            const portionHtml = food.portion ? `<span class="food-portion">(${food.portion})</span>` : '';
            li.innerHTML = `<span><strong>${food.name}</strong> ${portionHtml}</span> <span class="food-confidence">${food.confidence}</span>`;
            foodList.appendChild(li);
        });
    }

    // Populate Macros
    document.getElementById('cal-val').textContent = data.nutrition.calories;
    const weightEl = document.getElementById('weight-val');
    if (weightEl) {
        weightEl.textContent = data.nutrition.weight || '0g';
    }
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
