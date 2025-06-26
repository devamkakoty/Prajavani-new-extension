// --- IMPORTANT: REPLACE 'YOUR_GEMINI_API_KEY' WITH YOUR ACTUAL API KEY ---
// --- THIS IS INSECURE FOR A PUBLIC EXTENSION. USE A BACKEND FOR REAL APPS ---
const GEMINI_API_KEY = 'AIzaSyCZfFSUDrj5U8OP4aTxWKK2peh5Fsx-9pY';
const GEMINI_API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=${GEMINI_API_KEY}`;

async function getBase64FromImageUrl(url) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to fetch image: ${response.statusText}`);
    }
    const blob = await response.blob();
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result.split(',')[1]); // Get Base64 part
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  } catch (error) {
    console.error("Error fetching or converting image:", error);
    return null;
  }
}

async function getTranslationFromGemini(base64ImageData, mimeType = "image/jpeg") {
  if (!base64ImageData) return "Could not get image data.";

  const payload = {
    contents: [{
      parts: [
        { text: "Analyze this comic image. Identify any Kannada text within it and translate that Kannada text into English. Provide ONLY the English translation. If no Kannada text is found, respond with 'No Kannada text found.' Do not include any introductory phrases or explanations." },
        { inline_data: { mime_type: mimeType, data: base64ImageData } }
      ]
    }],
    // Optional: safetySettings and generationConfig can be added here
    // "generationConfig": {
    //   "temperature": 0.4,
    //   "topK": 32,
    //   "topP": 1,
    //   "maxOutputTokens": 1024,
    // }
  };

  try {
    const response = await fetch(GEMINI_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const errorData = await response.json();
      console.error("Gemini API Error:", errorData);
      return `Gemini API Error: ${errorData.error?.message || response.statusText}`;
    }

    const data = await response.json();
    if (data.candidates && data.candidates.length > 0 && data.candidates[0].content && data.candidates[0].content.parts && data.candidates[0].content.parts.length > 0) {
      let translation = data.candidates[0].content.parts[0].text.trim();
      // Clean up potential markdown or extra quotes
      if (translation.startsWith('"') && translation.endsWith('"')) {
        translation = translation.substring(1, translation.length - 1);
      }
      return translation;
    } else {
      console.warn("No translation found in Gemini response:", data);
      return "No translation found or unexpected response.";
    }
  } catch (error) {
    console.error("Error calling Gemini API:", error);
    return "Error connecting to translation service.";
  }
}

function displayTranslation(translationText, imageElementContainer) {
  // Remove any existing translation
  const existingTranslationDiv = imageElementContainer.querySelector('.comic-translation-pv');
  if (existingTranslationDiv) {
    existingTranslationDiv.remove();
  }

  const translationDiv = document.createElement('div');
  translationDiv.className = 'comic-translation-pv'; // Add a class for styling or re-selection
  translationDiv.style.border = '1px dashed #ccc';
  translationDiv.style.padding = '10px';
  translationDiv.style.marginTop = '10px';
  translationDiv.style.backgroundColor = '#f9f9f9';
  translationDiv.style.fontFamily = 'sans-serif';
  translationDiv.style.fontSize = '16px';
  translationDiv.style.whiteSpace = 'pre-wrap'; // Preserve line breaks from translation
  translationDiv.textContent = translationText;

  // Append after the figure element containing the picture
  imageElementContainer.insertAdjacentElement('afterend', translationDiv);
}

async function main() {
  console.log("Prajavani Comic Translator content script injected.");

  if (GEMINI_API_KEY === 'YOUR_GEMINI_API_KEY') {
    alert("Please set your Gemini API key in the content.js file of the extension.");
    console.error("Gemini API Key not set.");
    return;
  }

  // Selector based on your provided HTML structure
  // The image wrapper seems to be <div id="image-wrapper-story-page">
  // Inside it, a <figure> contains the <picture> which contains the <img>
  const imageWrapper = document.querySelector('div#image-wrapper-story-page');
  if (!imageWrapper) {
    console.log("Comic image wrapper not found on this page.");
    return;
  }

  const imgElement = imageWrapper.querySelector('figure picture img');
  const figureElement = imageWrapper.querySelector('figure'); // Element to append translation after

  if (!imgElement || !figureElement) {
    console.log("Comic image (img) or figure element not found.");
    return;
  }

  const imageUrl = imgElement.src;
  if (!imageUrl) {
    console.log("Image source URL not found.");
    return;
  }

  // Indicate processing
  displayTranslation("Translating, please wait...", figureElement);

  // Determine MIME type (Prajavani seems to use webp often, but jpg is also possible)
  let mimeType = "image/jpeg"; // Default
  if (imageUrl.toLowerCase().includes(".webp")) {
    mimeType = "image/webp";
  } else if (imageUrl.toLowerCase().includes(".png")) {
    mimeType = "image/png";
  }


  const base64ImageData = await getBase64FromImageUrl(imageUrl);
  if (!base64ImageData) {
     displayTranslation("Failed to load image data for translation.", figureElement);
     return;
  }

  const translation = await getTranslationFromGemini(base64ImageData, mimeType);
  displayTranslation(translation, figureElement);
}

// Ensure the script doesn't run multiple times if re-injected (though executeScript usually handles this)
if (!window.prajavaniComicTranslatorLoaded) {
  window.prajavaniComicTranslatorLoaded = true;
  main();
}