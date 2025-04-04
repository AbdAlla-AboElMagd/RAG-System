async function sendMessage() {
  const input = document.getElementById("user-input");
  const message = input.value.trim();
  const sendButton = document.getElementById("send-button");

  if (!message) return;

  // Disable button and show loading
  sendButton.disabled = true;
  showLoading();

  try {
    // Save message before clearing input
    const currentMessage = message;
    input.value = "";

    // Show user message immediately
    appendMessage(currentMessage, "user");

    console.log("Sending message:", currentMessage); // Debug log

    const response = await fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: currentMessage }),
    });

    console.log("Response status:", response.status); // Debug log

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log("Response data:", data); // Debug log

    if (data.error) {
      throw new Error(data.error);
    }

    // Add bot response
    appendMessage(data.answer, "bot", data.sources, data.is_local);
  } catch (error) {
    console.error("Chat error:", error);
    showToast(error.message, "danger");
    // Show error in chat
    appendMessage("Sorry, there was an error processing your message.", "bot");
  } finally {
    sendButton.disabled = false;
    hideLoading();
    input.focus();
  }
}

async function toggleFallback() {
  const toggle = document.getElementById("fallback-toggle");
  const indicator = document.getElementById("fallback-status");

  try {
    const response = await fetch("/toggle-fallback", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ enabled: toggle.checked }),
    });
    const data = await response.json();
    toggle.checked = data.enabled;

    // Show status toast
    showToast(
      `Global answers ${data.enabled ? "enabled" : "disabled"}`,
      data.enabled ? "info" : "warning"
    );
  } catch (error) {
    console.error("Error toggling fallback:", error);
    toggle.checked = !toggle.checked;
    showToast("Error toggling fallback setting", "danger");
  }
}

function appendMessage(message, type, sources = [], isLocal = true) {
  const chatContainer = document.getElementById("chat-container");
  if (!chatContainer) {
    console.error("Chat container not found");
    return;
  }

  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${type}-message animate__animated animate__fadeIn ${
    !isLocal ? "global" : ""
  }`;

  let content = `<div class="message-content">`;
  if (type === "bot") {
    if (!isLocal) {
      content += `<div class="global-indicator">
                <i class="bi bi-globe"></i> Global Response
            </div>`;
    }
    content += `<div class="bot-header">
            <i class="bi bi-robot"></i> <strong>Bot:</strong>
        </div>`;
  } else {
    content += `<div class="user-header">
            <i class="bi bi-person"></i> <strong>You:</strong>
        </div>`;
  }

  // Safely encode message content
  const encodedMessage = message.replace(/</g, "&lt;").replace(/>/g, "&gt;");
  content += `<div class="message-text">${encodedMessage}</div>`;

  if (sources && sources.length > 0) {
    content += `<div class="sources">
            <strong><i class="bi bi-journal-text"></i> Sources:</strong><br>
            ${sources
              .map(
                (s) => `<div class="source-item">
                <i class="bi bi-file-text"></i> ${s.source}: ${s.content}
            </div>`
              )
              .join("")}
        </div>`;
  }
  content += "</div>";

  messageDiv.innerHTML = content;
  chatContainer.appendChild(messageDiv);

  // Ensure smooth scroll to bottom with delay
  setTimeout(() => {
    messageDiv.scrollIntoView({
      behavior: "smooth",
      block: "end",
      inline: "nearest",
    });
  }, 100);

  // Ensure input section stays visible
  const inputSection = document.querySelector(".input-section");
  if (inputSection) {
    inputSection.style.display = "block";
  }
}

function showLoading() {
  document.getElementById("loading-overlay").style.display = "flex";
}

function hideLoading() {
  document.getElementById("loading-overlay").style.display = "none";
}

async function clearHistory() {
  try {
    await fetch("/clear", { method: "POST" });
    document.getElementById("chat-container").innerHTML = "";
    showToast("Chat history cleared successfully", "success");
  } catch (error) {
    console.error("Error clearing history:", error);
    showToast("Error clearing chat history", "danger");
  }
}

async function loadSources() {
  showLoading();
  try {
    const response = await fetch("/sources");
    const data = await response.json();
    const sourcesList = document.getElementById("sources-list");

    if (!sourcesList) {
      console.error("Sources list container not found");
      return;
    }

    sourcesList.innerHTML = "";

    if (data.sources && data.sources.length > 0) {
      data.sources.forEach((source) => {
        const div = document.createElement("div");
        div.className = "source-item animate__animated animate__fadeIn";
        div.innerHTML = `
          <div class="d-flex align-items-center">
            <i class="bi bi-file-pdf me-3 fs-4 text-primary"></i>
            <div>
              <h6 class="mb-1">${source.name}</h6>
              <small class="text-muted">Uploaded: ${new Date().toLocaleDateString()}</small>
            </div>
          </div>`;
        sourcesList.appendChild(div);
      });
    } else {
      sourcesList.innerHTML = `
        <div class="alert alert-info">
          <div class="d-flex align-items-center gap-2">
            <i class="bi bi-info-circle fs-4"></i>
            <span>No documents available. Upload a PDF to get started.</span>
          </div>
        </div>`;
    }
  } catch (error) {
    console.error("Error loading sources:", error);
    showToast("Error loading sources", "danger");
  } finally {
    hideLoading();
  }
}

async function uploadPDF() {
  const fileInput = document.getElementById("pdf-upload");
  const file = fileInput.files[0];
  if (!file) {
    showToast("Please select a file first", "warning");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  const uploadButton = document.getElementById("upload-button");
  uploadButton.disabled = true;
  showLoading();

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (response.ok) {
      showToast("File uploaded successfully", "success");
      loadSources();
      fileInput.value = "";
    } else {
      showToast(`Error: ${data.error}`, "danger");
    }
  } catch (error) {
    showToast("Error uploading file", "danger");
    console.error("Error:", error);
  } finally {
    uploadButton.disabled = false;
    hideLoading();
  }
}

function showToast(message, type = "info") {
  const toastContainer = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = `toast align-items-center text-white bg-${type} border-0`;
  toast.setAttribute("role", "alert");
  toast.setAttribute("aria-live", "assertive");
  toast.setAttribute("aria-atomic", "true");

  toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast" aria-label="Close"></button>
        </div>
    `;

  toastContainer.appendChild(toast);
  const bsToast = new bootstrap.Toast(toast);
  bsToast.show();

  toast.addEventListener("hidden.bs.toast", () => {
    toast.remove();
  });
}

// Event Listeners
document.addEventListener("DOMContentLoaded", () => {
  // Make sure elements exist before adding listeners
  const userInput = document.getElementById("user-input");
  const sendButton = document.getElementById("send-button");
  const chatContainer = document.getElementById("chat-container");

  if (!userInput || !sendButton || !chatContainer) {
    console.error("Required elements not found");
    return;
  }

  // Reset event listeners
  const newSendButton = sendButton.cloneNode(true);
  sendButton.parentNode.replaceChild(newSendButton, sendButton);

  // Add event listeners
  userInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  newSendButton.addEventListener("click", sendMessage);

  // Focus input on load
  userInput.focus();

  // Initialize sources
  loadSources();

  // Enhanced tab switching handlers
  const chatTab = document.getElementById("chat-tab");
  const sourcesTab = document.getElementById("sources-tab");
  const chatPane = document.getElementById("Chat");
  const sourcesPane = document.getElementById("Sources");

  function showTab(activePane, inactivePane) {
    // Remove all classes first
    inactivePane.classList.remove("show", "active");
    activePane.classList.remove("show", "active");

    // Force hide inactive tab
    inactivePane.style.cssText = "display: none !important";

    // Show active tab
    activePane.classList.add("show", "active");
    activePane.style.cssText = "display: flex !important";

    // Additional cleanup
    if (activePane.id === "Chat") {
      document.getElementById("Sources").style.cssText =
        "display: none !important";
    } else {
      document.getElementById("Chat").style.cssText =
        "display: none !important";
    }
  }

  chatTab.addEventListener("show.bs.tab", () => {
    showTab(chatPane, sourcesPane);

    // Scroll chat to bottom
    const chatContainer = document.getElementById("chat-container");
    if (chatContainer) {
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  });

  sourcesTab.addEventListener("show.bs.tab", () => {
    showTab(sourcesPane, chatPane);
    loadSources();

    // Scroll sources to top
    const sourcesContainer = document.querySelector(".sources-container");
    if (sourcesContainer) {
      sourcesContainer.scrollTop = 0;
    }
  });

  // Initialize sources on first load
  loadSources();
});

// Add resize handler
window.addEventListener("resize", () => {
  const chatContainer = document.getElementById("chat-container");
  if (chatContainer) {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }
});
