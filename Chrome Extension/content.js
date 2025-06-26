// This is your new, updated content.js file

// The URL for your backend must be correct
const FLASK_SERVICE_URL = 'http://localhost:5000';

/**
 * A helper function to make API calls to the Flask backend.
 * @param {string} endpoint The endpoint to hit (e.g., '/translate-comic').
 * @param {object} payload The JSON payload to send.
 * @returns {Promise<object>} The JSON response from the server.
 */
async function callFlaskService(endpoint, payload) {
    try {
        const response = await fetch(`${FLASK_SERVICE_URL}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            console.error(`Service Error at ${endpoint}:`, errorData);
            throw new Error(`Service Error: ${errorData.error || response.statusText}`);
        }
        return await response.json();

    } catch (error) {
        console.error(`Error calling service at ${endpoint}:`, error);
        throw error;
    }
}

/**
 * This is the new function that replaces the Google Translate Element.
 * It manually finds text nodes, gets them translated via your backend,
 * and replaces them on the page.
 */
async function translatePageTextManually() {
    console.log("Starting manual page text translation...");
    const TEXT_NODE_TYPE = 3; // Node.TEXT_NODE constant
    const textNodes = [];

    // Use a TreeWalker to efficiently find all non-empty text nodes in the document body
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, {
        acceptNode: (node) => {
            // Reject nodes that are inside script/style tags or are just whitespace
            if (node.parentElement.tagName.match(/SCRIPT|STYLE/i) || !node.nodeValue.trim()) {
                return NodeFilter.FILTER_REJECT;
            }
            return NodeFilter.FILTER_ACCEPT;
        }
    });

    // Populate the list of text nodes
    while (walker.nextNode()) {
        textNodes.push(walker.currentNode);
    }

    if (textNodes.length === 0) {
        console.log("No translatable text found on the page.");
        return;
    }

    console.log(`Found ${textNodes.length} text nodes to translate.`);
    const originalTexts = textNodes.map(node => node.nodeValue);

    try {
        // Call your new backend endpoint to translate the batch of texts
        const result = await callFlaskService('/translate-text-batch', {
            texts: originalTexts,
            source_lang: 'kn',
            target_lang: 'en'
        });

        if (result.success && result.translations) {
            // Replace the original text with the translation
            textNodes.forEach((node, index) => {
                if (result.translations[index]) {
                    node.nodeValue = result.translations[index];
                }
            });
            console.log("✅ Page text translation completed successfully.");
        }
    } catch (error) {
        console.error("Could not complete text translation:", error);
    }
}


/**
 * Processes a single image to find text and replace it with a translated overlay.
 * (This is your existing image processing logic, slightly refactored for clarity)
 */
async function processImage(imgElement) {
    console.log('Processing image:', imgElement.src);
    const containerElement = imgElement.closest('figure') || imgElement.parentElement;
    if (!containerElement) {
        console.warn("Could not find a container for image:", imgElement.src);
        return;
    }

    // Display a "translating..." message for this specific image
    const tempMessageDiv = displayMessage(`🔄 Translating image...`, containerElement);

    try {
        const result = await callFlaskService('/translate-comic', {
            image_url: imgElement.src,
            source_lang: 'kn',
            target_lang: 'en'
        });

        // Remove the "translating..." message
        tempMessageDiv.remove();

        // Display the final result (overlay image or an error)
        // ONLY display the result if the translation was successful AND we have an overlay.
        // If there's no text, result.overlay_image will be missing, and this block is skipped.
     if (result.success && result.overlay_image) {
        displayTranslationResult(result, imgElement, containerElement);
     }

    } catch (error) {
        tempMessageDiv.textContent = `❌ Error translating image: ${error.message}`;
    }
}

/**
 * Displays the final translated image overlay.
 */
function displayTranslationResult(result, imageElement, containerElement) {
    const existingOverlay = containerElement.nextElementSibling;
    if (existingOverlay && existingOverlay.classList.contains('comic-translation-pv')) {
        existingOverlay.remove();
    }

    if (result.success && result.overlay_image) {
        const overlayContainer = document.createElement('div');
        // ... (all your styling for the overlay container) ...
        overlayContainer.className = 'comic-translation-pv';
        overlayContainer.style.marginTop = '10px';
        overlayContainer.innerHTML = `
            <div style="font-weight: bold; color: #2e7d32; margin-bottom: 5px;">✅ Translated Image</div>
            <img src="data:image/jpeg;base64,${result.overlay_image}" style="width: 100%; height: auto; border-radius: 4px;" />
        `;
        containerElement.insertAdjacentElement('afterend', overlayContainer);
    } else {
        displayMessage(result.message || "No text found in image.", containerElement);
    }
}

/**
 * A helper to display various status messages.
 */
function displayMessage(message, containerElement) {
    const existingMessage = containerElement.nextElementSibling;
    if (existingMessage && existingMessage.classList.contains('comic-translation-pv')) {
        existingMessage.remove();
    }

    const messageDiv = document.createElement('div');
    messageDiv.className = 'comic-translation-pv';
    messageDiv.style.padding = '10px';
    messageDiv.style.marginTop = '10px';
    messageDiv.style.border = '1px dashed #ccc';
    messageDiv.textContent = message;
    containerElement.insertAdjacentElement('afterend', messageDiv);
    return messageDiv;
}


/**
 * Main function to find and translate all content on the page.
 */
async function main() {
    console.log("Prajavani Content Translator Activated");

    // --- STEP 1: Translate all text on the page using the new manual method ---
    await translatePageTextManually();

    // --- STEP 2: Find and translate all images (your existing logic) ---
    const allImages = document.querySelectorAll('img');
    const MIN_DIMENSION = 200;
    let processedImageCount = 0;

    for (const img of allImages) {
        if (img.naturalWidth > MIN_DIMENSION && img.naturalHeight > MIN_DIMENSION) {
            processedImageCount++;
            // Process images one by one
            await processImage(img);
        }
    }

    if (processedImageCount === 0) {
        console.log(`No images larger than ${MIN_DIMENSION}px found to translate.`);
    } else {
        console.log(`Finished processing ${processedImageCount} images.`);
    }
}

// --- Entry Point ---
if (!window.prajavaniTranslatorLoaded) {
    window.prajavaniTranslatorLoaded = true;
    main();
}