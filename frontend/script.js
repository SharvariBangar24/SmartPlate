// frontend/script.js

// ============================================================
// DOM Elements
// ============================================================
const scanBtnHero = document.getElementById('scan-btn');
const navGetStarted = document.getElementById('nav-get-started');
const scannerSection = document.getElementById('scanner');
const uploadImageInput = document.getElementById('upload-image');
const takePhotoInput = document.getElementById('take-photo');
const imagePreview = document.getElementById('image-preview');
const placeholderText = document.getElementById('placeholder-text');
const previewBox = document.getElementById('preview-box');
const analyzeBtn = document.getElementById('analyze-btn');
const imageInfoStrip = document.getElementById('image-info-strip');
const fileNameDisplay = document.getElementById('file-name-display');
const changeImageBtn = document.getElementById('change-image-btn');

const loadingSection = document.getElementById('loading-section');
const errorSection = document.getElementById('error-section');
const errorMessage = document.getElementById('error-message');
const retryBtn = document.getElementById('retry-btn');
const resultsSection = document.getElementById('results');
const resetBtn = document.getElementById('reset-btn');

// The currently selected file (preserved for re-analysis / retry)
let currentFile = null;

// ============================================================
// API Configuration
// ============================================================
const API_BASE = 'http://127.0.0.1:5000';

// ============================================================
// Event Listeners
// ============================================================
document.addEventListener('DOMContentLoaded', () => {

    // Smooth scroll to scanner section when hero button is clicked
    scanBtnHero.addEventListener('click', () => {
        scannerSection.scrollIntoView({ behavior: 'smooth' });
    });

    // "Get Started" nav button
    if (navGetStarted) {
        navGetStarted.addEventListener('click', () => {
            scannerSection.scrollIntoView({ behavior: 'smooth' });
        });
    }

    // Listen for file selection on "Upload Image"
    uploadImageInput.addEventListener('change', handleImageSelection);

    // Listen for file selection on "Take Photo"
    takePhotoInput.addEventListener('change', handleImageSelection);

    // Perform analysis when button is clicked
    analyzeBtn.addEventListener('click', performAnalysis);

    // Reset app for a new scan
    resetBtn.addEventListener('click', resetApp);

    // Retry after error
    if (retryBtn) {
        retryBtn.addEventListener('click', () => {
            errorSection.hidden = true;
            if (currentFile) {
                performAnalysis();
            } else {
                scannerSection.hidden = false;
                scannerSection.scrollIntoView({ behavior: 'smooth' });
            }
        });
    }

    // Change image button
    if (changeImageBtn) {
        changeImageBtn.addEventListener('click', () => {
            uploadImageInput.click();
        });
    }
});

// ============================================================
// Image Selection
// ============================================================

/**
 * Reads the selected image file and displays it in the preview area
 */
function handleImageSelection(event) {
    const file = event.target.files[0];
    if (file) {
        currentFile = file;

        const reader = new FileReader();

        reader.onload = function (e) {
            // Display the image
            imagePreview.src = e.target.result;
            imagePreview.hidden = false;

            // Hide the placeholder text
            placeholderText.hidden = true;

            // Add visual state to preview box
            if (previewBox) {
                previewBox.classList.add('has-image');
            }

            // Show the image info strip
            if (imageInfoStrip) {
                imageInfoStrip.hidden = false;
                fileNameDisplay.textContent = file.name;
            }

            // Show the analyze button
            analyzeBtn.hidden = false;
        }

        // Read the image file as a Data URL (base64 string)
        reader.readAsDataURL(file);
    }
}

// ============================================================
// Analysis Pipeline
// ============================================================

/**
 * Simulates the analysis step UI progression
 */
function setAnalysisStep(stepName, state) {
    const stepsContainer = document.getElementById('analysis-steps');
    if (!stepsContainer) return;

    const allSteps = stepsContainer.querySelectorAll('.analysis-step');
    allSteps.forEach(step => {
        const sName = step.getAttribute('data-step');
        if (sName === stepName) {
            step.classList.remove('active', 'done');
            step.classList.add(state);
        }
    });
}

function resetAnalysisSteps() {
    const stepsContainer = document.getElementById('analysis-steps');
    if (!stepsContainer) return;

    const allSteps = stepsContainer.querySelectorAll('.analysis-step');
    allSteps.forEach(step => {
        step.classList.remove('active', 'done');
    });
}

/**
 * Performs the actual API call to the backend.
 * Currently calls /api/test. When the full pipeline is ready,
 * this should call /api/analyze with the image FormData.
 */
async function performAnalysis() {
    // Hide scanner and error, show loading spinner
    scannerSection.hidden = true;
    errorSection.hidden = true;
    resultsSection.hidden = true;
    loadingSection.hidden = false;

    // Scroll to loading area
    loadingSection.scrollIntoView({ behavior: 'smooth' });

    // Reset and animate analysis steps
    resetAnalysisSteps();

    // Animate steps with delays for visual feedback
    const steps = ['upload', 'segment', 'classify', 'nutrition'];
    for (let i = 0; i < steps.length; i++) {
        await delay(400);
        // Mark previous step as done
        if (i > 0) {
            setAnalysisStep(steps[i - 1], 'done');
        }
        setAnalysisStep(steps[i], 'active');
    }

    try {
        const response = await fetch(`${API_BASE}/api/test`);

        if (!response.ok) {
            throw new Error(`Server responded with status ${response.status}`);
        }

        const data = await response.json();

        console.log("Backend response:", data);

        // Mark last step as done
        setAnalysisStep('nutrition', 'done');
        await delay(300);

        // Build results from backend data
        // For now, use the backend response as the insight.
        // When the full pipeline is implemented, parse real
        // food items, confidence scores, and nutrition data.
        const resultData = {
            foods: [
                { name: "Backend Connected", confidence: "100%" }
            ],
            nutrition: {
                calories: "--",
                protein: "--",
                carbs: "--",
                fat: "--"
            },
            insight: data.message || "Analysis complete."
        };

        populateResults(resultData);

        loadingSection.hidden = true;
        resultsSection.hidden = false;
        resultsSection.scrollIntoView({ behavior: 'smooth' });

    } catch (error) {
        console.error("Backend connection error:", error);

        loadingSection.hidden = true;
        showError("Could not connect to the Smart Plate backend. Please ensure the Flask server is running at " + API_BASE);
    }
}

// ============================================================
// Results Population
// ============================================================

/**
 * Populates the HTML dashboard with the provided data.
 *
 * Expected data shape:
 *   {
 *     foods: [{ name: string, confidence: string, region?: string }],
 *     nutrition: { calories, protein, carbs, fat },
 *     insight: string
 *   }
 */
function populateResults(data) {
    // ---- Detected Foods List ----
    const foodList = document.getElementById('food-list');
    foodList.innerHTML = '';

    data.foods.forEach((food, index) => {
        const li = document.createElement('li');

        // Determine confidence level for badge color
        const confNum = parseFloat(food.confidence);
        let confClass = 'high';
        if (!isNaN(confNum)) {
            if (confNum < 70) confClass = 'low';
            else if (confNum < 85) confClass = 'medium';
        }

        li.innerHTML = `
            <span class="food-index">${index + 1}</span>
            <span class="food-details">
                <span class="food-name">${food.name}</span>
                ${food.region ? `<span class="food-meta">Region: ${food.region}</span>` : ''}
            </span>
            <span class="food-confidence ${confClass}">${food.confidence}</span>
        `;
        foodList.appendChild(li);
    });

    // ---- Macros ----
    const cal = data.nutrition.calories;
    const pro = data.nutrition.protein;
    const carb = data.nutrition.carbs;
    const fat = data.nutrition.fat;

    document.getElementById('cal-val').textContent = typeof cal === 'number' ? cal : cal;
    document.getElementById('pro-val').textContent = typeof pro === 'number' ? pro + 'g' : pro;
    document.getElementById('carb-val').textContent = typeof carb === 'number' ? carb + 'g' : carb;
    document.getElementById('fat-val').textContent = typeof fat === 'number' ? fat + 'g' : fat;

    // ---- Total Plate Nutrition ----
    document.getElementById('total-cal').textContent = typeof cal === 'number' ? cal + ' kcal' : '—';
    document.getElementById('total-pro').textContent = typeof pro === 'number' ? pro + ' g' : '—';
    document.getElementById('total-carb').textContent = typeof carb === 'number' ? carb + ' g' : '—';
    document.getElementById('total-fat').textContent = typeof fat === 'number' ? fat + ' g' : '—';

    // Compute total energy if numeric
    if (typeof cal === 'number') {
        document.getElementById('total-energy').textContent = cal + ' kcal';
    } else {
        document.getElementById('total-energy').textContent = '—';
    }

    // ---- Health Insight ----
    document.getElementById('health-insight-text').textContent = data.insight || 'No additional insights available.';
}

// ============================================================
// Error Handling
// ============================================================

function showError(message) {
    errorSection.hidden = false;
    errorMessage.textContent = message;
    errorSection.scrollIntoView({ behavior: 'smooth' });
}

// ============================================================
// Reset
// ============================================================

/**
 * Resets the application state back to the beginning
 */
function resetApp() {
    // Clear file inputs so the same file can be selected again if needed
    uploadImageInput.value = '';
    takePhotoInput.value = '';
    currentFile = null;

    // Reset preview area
    imagePreview.src = '';
    imagePreview.hidden = true;
    placeholderText.hidden = false;
    analyzeBtn.hidden = true;

    if (previewBox) {
        previewBox.classList.remove('has-image');
    }

    if (imageInfoStrip) {
        imageInfoStrip.hidden = true;
    }

    // Hide results and errors, show scanner again
    resultsSection.hidden = true;
    errorSection.hidden = true;
    scannerSection.hidden = false;

    // Reset analysis step indicators
    resetAnalysisSteps();

    // Scroll back up to the scanner section
    scannerSection.scrollIntoView({ behavior: 'smooth' });
}

// ============================================================
// Utilities
// ============================================================

function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
