// This function will be executed when the extension icon is clicked
async function translateComicOnPage(tab) {
  if (tab.url){ // && tab.url.includes("prajavani.net/cartoons")) {
    try {
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ["content.js"],
      });
    } catch (err) {
      console.error("Failed to inject content script:", err);
    }
  } else {
    console.log("Not a Prajavani cartoon page.");
  }
}

chrome.action.onClicked.addListener((tab) => {
  translateComicOnPage(tab);
});