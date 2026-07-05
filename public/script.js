/**
 * Squatch Neuro Hub
 * Neural-linked AI assistant with associative cluster memory and multi-provider support
 */

// DOM Elements
const providerSelect = document.getElementById('providerSelect');
const modelSelect = document.getElementById('modelSelect');
const newChatBtn = document.getElementById('newChatBtn');
const chatContainer = document.getElementById('chatContainer');
const messagesContainer = document.getElementById('messages');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const typingIndicator = document.getElementById('typingIndicator');
const micBtn = document.getElementById('micBtn');
const convoModeBtn = document.getElementById('convoModeBtn');
const speakerToggle = document.getElementById('speakerToggle');
const superSearchBtn = document.getElementById('superSearchBtn');
const settingsBtn = document.getElementById('settingsBtn');
const settingsModal = document.getElementById('settingsModal');
const closeModal = document.getElementById('closeModal');
const saveSettingsBtn = document.getElementById('saveSettings');

// SquatchServe model status elements
const modelStatusBar = document.getElementById('modelStatusBar');
const modelStatusText = document.getElementById('modelStatusText');
const unloadModelBtn = document.getElementById('unloadModelBtn');

// State variables
let currentProvider = '';
let currentModel = '';
let providers = [];
let conversation = [];
let isTyping = false;
let squatchserveStatusInterval = null;
let loadedSquatchserveModel = null;
let lastAssistantMessageId = null;
let streamingMessageElement = null;
let pendingContent = null;
let animationFrameId = null;
let ttsEnabled = false;
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];
let conversationMode = false;
let silenceTimer = null;
let audioContext = null;
let analyser = null;
let audioUnlocked = false;
let superSearch = false;
const TTS_URL = '/api/tts';
const STT_URL = '/api/stt';

// TTS chunked pipeline — sends text to TTS as sentences complete during streaming
const ttsChunker = {
  sentIndex: 0,
  chunkIndex: 0,
  active: false,
  flushed: false,
  queue: [],          // { index, state: 'loading'|'ready'|'error', audio, url }
  fetchChain: Promise.resolve(),  // sequential fetch pipeline
  playing: false,
  currentAudio: null,
  abortController: null,

  start() {
    console.log('[ttsChunker] start() called, ttsEnabled:', ttsEnabled);
    this.cancel();
    this.sentIndex = 0;
    this.chunkIndex = 0;
    this.active = true;
    this.flushed = false;
    this.queue = [];
    this.fetchChain = Promise.resolve();
    this.playing = false;
    this.currentAudio = null;
    this.abortController = new AbortController();
  },

  feed(fullText) {
    if (!this.active) {
      console.log('[ttsChunker] feed() skipped — not active');
      return;
    }

    while (true) {
      const unsent = fullText.slice(this.sentIndex);
      if (unsent.length < 80) break;

      // Find first sentence boundary (. ! ? followed by whitespace, or paragraph break) at >= 80 chars
      const re = /[.!?](?=\s)|\n\n+/g;
      let found = false;
      let m;
      while ((m = re.exec(unsent)) !== null) {
        const textEnd = m[0].startsWith('\n') ? m.index : m.index + 1;
        if (textEnd >= 80) {
          const chunk = unsent.slice(0, textEnd).trim();
          // Advance past the boundary and any trailing whitespace
          this.sentIndex += m.index + m[0].length;
          if (chunk) this._sendChunk(chunk);
          found = true;
          break;
        }
      }
      if (!found) break;
    }
  },

  flush(fullText) {
    console.log('[ttsChunker] flush() called, active:', this.active, 'sentIndex:', this.sentIndex, 'fullText length:', fullText.length);
    if (!this.active) return;
    const remaining = fullText.slice(this.sentIndex).trim();
    console.log('[ttsChunker] flush() remaining text length:', remaining.length);
    if (remaining) {
      this._sendChunk(remaining);
    }
    this.flushed = true;
    this.active = false;
    // If no chunks were queued at all, trigger conversation mode immediately
    if (this.queue.length === 0 && conversationMode && !isRecording) {
      startRecording();
    }
  },

  cancel() {
    this.active = false;
    this.flushed = false;
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
    if (this.currentAudio) {
      this.currentAudio.pause();
      this.currentAudio.currentTime = 0;
      this.currentAudio = null;
    }
    for (const entry of this.queue) {
      if (entry.url) URL.revokeObjectURL(entry.url);
    }
    this.queue = [];
    this.fetchChain = Promise.resolve();
    this.playing = false;
    this.sentIndex = 0;
    this.chunkIndex = 0;
  },

  _sendChunk(text) {
    const index = this.chunkIndex++;
    console.log(`[ttsChunker] _sendChunk(${index}) text length:`, text.length, 'preview:', text.substring(0, 60));

    // Clean markdown for TTS
    text = text
      .replace(/\*\*(.*?)\*\*/g, '$1')
      .replace(/\*(.*?)\*/g, '$1')
      .replace(/`(.*?)`/g, '$1')
      .replace(/#{1,6}\s?/g, '')
      .replace(/\n/g, ' ')
      .replace(/\s+/g, ' ')
      .trim();

    if (!text) return;

    const entry = { index, state: 'loading', audio: null, url: null };
    this.queue.push(entry);

    // Chain fetches sequentially: each chunk waits for the previous fetch to complete.
    // This gives 1-chunk lookahead — while chunk N plays, chunk N+1 fetch is in flight.
    this.fetchChain = this.fetchChain.then(() => this._fetchChunk(entry, text, index));
  },

  _fetchChunk(entry, text, index) {
    const signal = this.abortController?.signal;
    return fetch(TTS_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
      signal
    })
    .then(res => {
      console.log(`[ttsChunker] chunk ${index} TTS response:`, res.status);
      if (!res.ok) throw new Error(`TTS ${res.status}`);
      return res.blob();
    })
    .then(blob => {
      console.log(`[ttsChunker] chunk ${index} audio blob size:`, blob.size);
      if (blob.size === 0) throw new Error('Empty audio');
      entry.url = URL.createObjectURL(blob);
      entry.audio = new Audio(entry.url);
      entry.state = 'ready';
      this._playNext();
    })
    .catch(err => {
      if (err.name === 'AbortError') return;
      console.error(`TTS chunk ${index} error:`, err);
      entry.state = 'error';
      this._playNext();
    });
  },

  _playNext() {
    if (this.playing || this.queue.length === 0) {
      // Check if all done
      if (!this.playing && this.queue.length === 0 && this.flushed) {
        if (conversationMode && !isRecording) {
          startRecording();
        }
      }
      return;
    }

    const next = this.queue[0];

    // Wait for in-order playback — if first chunk is still loading, wait
    if (next.state === 'loading') return;

    if (next.state === 'error') {
      this.queue.shift();
      this._playNext();
      return;
    }

    // state === 'ready'
    this.playing = true;
    this.currentAudio = next.audio;

    // Guard against double-fire (play() rejection + onerror can both trigger)
    const cleanup = () => {
      if (!this.playing) return;
      next.audio.onended = null;
      next.audio.onerror = null;
      URL.revokeObjectURL(next.url);
      this.queue.shift();
      this.playing = false;
      this.currentAudio = null;
      this._playNext();
    };

    next.audio.onended = cleanup;
    next.audio.onerror = cleanup;

    next.audio.play().catch(e => {
      console.error('TTS chunk play failed:', e);
      cleanup();
    });
  }
};

// Conversation history state
let currentConversationId = null;
let conversations = [];
let sidebarCollapsed = false;   

// Initialize the app
document.addEventListener('DOMContentLoaded', () => {
  loadProviders();
  loadConversations();
  setupEventListeners();
  setupSidebarListeners();
  loadSettings();
  checkMobileView();
});

// Load available providers
async function loadProviders() {
  try {
    const hasClaudeKey = !!localStorage.getItem('claudeApiKey');
    const hasGrokKey = !!localStorage.getItem('grokApiKey');
    const hasOpenAIKey = !!localStorage.getItem('openaiApiKey');

    const response = await fetch('/api/providers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ hasClaudeKey, hasGrokKey, hasOpenAIKey })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    providers = data.providers || [];

    providerSelect.innerHTML = '<option value="">Select a provider</option>';
    providers.forEach(provider => {
      const option = document.createElement('option');
      option.value = provider.id;
      // Add visual indicator if API key is missing for providers that require it
      const needsKey = provider.requiresKey && !provider.hasKey;
      option.textContent = needsKey ? `${provider.name} (API key required)` : provider.name;
      option.dataset.requiresKey = provider.requiresKey;
      option.dataset.hasKey = provider.hasKey;
      providerSelect.appendChild(option);
    });

    // Restore saved provider
    const savedProvider = localStorage.getItem('selectedProvider');
    if (savedProvider) {
      providerSelect.value = savedProvider;
      currentProvider = savedProvider;
      await loadModelsForProvider(savedProvider);

      // Initialize model status bar for SquatchServe
      updateModelStatusBarVisibility();
    }
  } catch (error) {
    console.error('Error loading providers:', error);
    addMessage('error', 'Failed to load providers. Please check your connection.');
  }
}

// Load models for selected provider
async function loadModelsForProvider(providerId) {
  try {
    modelSelect.innerHTML = '<option value="">Select a model</option>';

    const provider = providers.find(p => p.id === providerId);
    if (!provider) return;

    // Check if API key is required but missing
    if (provider.requiresKey && !provider.hasKey) {
      const keyName = provider.type === 'claude' ? 'Claude' : provider.type === 'openai' ? 'OpenAI' : 'Grok';
      addMessage('system', `${keyName} requires an API key. Click the ⚙️ Settings button to add your API key.`);
      return;
    }

    let models = [];

    // Instance-based providers (ollama, vllm, llamacpp) use the instance/models endpoint
    if (provider.instanceName) {
      const response = await fetch('/api/instance/models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ providerType: provider.type, instanceName: provider.instanceName })
      });
      if (!response.ok) throw new Error(`${provider.name} not available`);
      const data = await response.json();
      models = data.models || [];
    } else if (provider.type === 'openai') {
      // Fetch OpenAI models dynamically
      const apiKey = localStorage.getItem('openaiApiKey');
      const response = await fetch('/api/openai/models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ apiKey })
      });
      if (!response.ok) throw new Error('Failed to fetch OpenAI models');
      const data = await response.json();
      models = data.models || [];
    } else if (provider.type === 'squatchserve') {
      // Fetch SquatchServe models dynamically
      const squatchserveHost = localStorage.getItem('squatchserveHost') || '';
      const url = squatchserveHost
        ? `/api/squatchserve/models?host=${encodeURIComponent(squatchserveHost)}`
        : '/api/squatchserve/models';
      const response = await fetch(url);
      if (!response.ok) throw new Error('SquatchServe not available');
      const data = await response.json();
      models = data.models || [];
    } else {
      // Use pre-defined models (Claude, Grok)
      models = provider.models || [];
    }

    models.forEach(model => {
      const option = document.createElement('option');
      option.value = model.id;
      option.textContent = model.name;
      modelSelect.appendChild(option);
    });

    // Restore saved model
    const savedModel = localStorage.getItem('selectedModel');
    if (savedModel) {
      modelSelect.value = savedModel;
      currentModel = savedModel;
    }
  } catch (error) {
    console.error('Error loading models:', error);
    addMessage('error', `Failed to load models for ${providerId}. Check that the server is running.`);
  }
}

// Load conversation from session storage
function loadConversation() {
  try {
    const savedConversation = sessionStorage.getItem('ollamaChatConversation');
    if (savedConversation) {
      conversation = JSON.parse(savedConversation);
      renderMessages();
    } else {
      // Show welcome message
      addMessage('system', 'Welcome to Neuro Hub! Select a provider and model to start chatting.');
    }
  } catch (error) {
    console.error('Error loading conversation:', error);
    conversation = [];
    addMessage('system', 'Welcome to Neuro Hub! Select a provider and model to start chatting.');
  }
}

// Save conversation to session storage
function saveConversation() {
  try {
    sessionStorage.setItem('ollamaChatConversation', JSON.stringify(conversation));
  } catch (error) {
    console.error('Error saving conversation:', error);
  }
}

// Set up event listeners
function setupEventListeners() {
  providerSelect.addEventListener('change', handleProviderChange);
  modelSelect.addEventListener('change', handleModelChange);
  newChatBtn.addEventListener('click', newChat);
  sendBtn.addEventListener('click', sendMessage);
  messageInput.addEventListener('keydown', handleKeyDown);
  messageInput.addEventListener('input', autoResizeInput);
  micBtn.addEventListener('mousedown', startRecording);
  convoModeBtn.addEventListener('click', toggleConversationMode);
  micBtn.addEventListener('mouseup', stopRecording);
  micBtn.addEventListener('mouseleave', stopRecording);
  speakerToggle.addEventListener('click', toggleTTS);
  superSearchBtn.addEventListener('click', toggleSuperSearch);
  settingsBtn.addEventListener('click', openSettings);
  closeModal.addEventListener('click', closeSettings);
  saveSettingsBtn.addEventListener('click', saveSettingsHandler);

  // Settings nav tab switching
  document.querySelectorAll('.settings-nav-item').forEach(tab => {
    tab.addEventListener('click', () => switchSettingsTab(tab.dataset.settingsTab));
  });

  // SquatchServe unload button
  if (unloadModelBtn) {
    unloadModelBtn.addEventListener('click', unloadSquatchserveModel);
  }

  // Close modal on outside click
  settingsModal.addEventListener('click', (e) => {
    if (e.target === settingsModal) {
      closeSettings();
    }
  });
}

// Handle provider selection
async function handleProviderChange() {
  currentProvider = providerSelect.value;
  localStorage.setItem('selectedProvider', currentProvider);

  // Update model status bar visibility for SquatchServe
  updateModelStatusBarVisibility();

  if (currentProvider) {
    await loadModelsForProvider(currentProvider);
    addMessage('system', `Provider selected: ${providerSelect.options[providerSelect.selectedIndex].text}`);
  }
}

// Handle model selection
function handleModelChange() {
  currentModel = modelSelect.value;
  localStorage.setItem('selectedModel', currentModel);
  
  if (currentModel) {
    addMessage('system', `Model selected: ${currentModel}`);
  }
}

// Handle sending a message
async function sendMessage() {
  const message = messageInput.value.trim();
  // Cancel any pending TTS from previous message
  ttsChunker.cancel();
  if (!message || !currentModel || !currentProvider) {
    if (!currentProvider || !currentModel) {
      addMessage('error', 'Please select a provider and model first.');
    }
    return;
  }

  // Fix 6: Disable send button during streaming
  sendBtn.disabled = true;
  sendBtn.classList.add('btn-disabled');

  // Add user message to conversation
  addMessage('user', message);
  messageInput.value = '';
  autoResizeInput();

  // Show typing indicator
  showTypingIndicator();

  try {
    let response = null;
    const conversationMessages = conversation
      .filter(msg => msg.role === 'user' || msg.role === 'assistant')
      .map(msg => ({
        role: msg.role,
        content: msg.content
      }));

    // Resolve provider type and instance name from current selection
    const selectedProvider = providers.find(p => p.id === currentProvider);
    const providerType = selectedProvider?.type || currentProvider;
    const instanceName = selectedProvider?.instanceName || undefined;

    // Use memory-enhanced chat endpoint
    const ollamaHost = localStorage.getItem('ollamaHost') || undefined;
    const squatchserveHost = localStorage.getItem('squatchserveHost') || undefined;
    const llamacppHost = localStorage.getItem('llamacppHost') || undefined;
    const apiKey = providerType === 'claude'
      ? localStorage.getItem('claudeApiKey')
      : providerType === 'openai'
        ? localStorage.getItem('openaiApiKey')
        : providerType === 'grok'
          ? localStorage.getItem('grokApiKey')
          : undefined;

    const requestBody = {
      model: currentModel,
      messages: conversationMessages,
      provider: providerType,
      instanceName,
      conversation_id: currentConversationId,
      ollamaHost,
      squatchserveHost,
      llamacppHost,
      apiKey,
      searxngHost: localStorage.getItem('searxngHost') || undefined,
      ttsEnabled,
      superSearch
    };
    console.log('[sendMessage] Provider:', providerType, 'Instance:', instanceName);

    response = await fetch('/api/chat/memory', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestBody)
    });

    // SECURITY FIX: Check if response exists before accessing properties
    if (!response) {
      throw new Error('No response received from provider');
    }

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
    }

    // Get conversation ID from response headers
    const newConversationId = response.headers.get('X-Conversation-Id');
    const hasMemoryContext = response.headers.get('X-Has-Memory-Context') === 'true';

    if (newConversationId && !currentConversationId) {
      currentConversationId = newConversationId;
    }

    // Show memory context indicator if applicable
    if (hasMemoryContext) {
      showMemoryIndicator(response);
    }

    // Show tools indicator if applicable
    const usedTools = response.headers.get('X-Tools-Used') === 'true';
    if (usedTools) {
      showToolsIndicator();
    }

    // Handle streaming response
    let fullResponse = '';
    const assistantMessageId = Date.now();
    conversation.push({
      role: 'assistant',
      content: '',
      id: assistantMessageId
    });
    lastAssistantMessageId = assistantMessageId;

    // Create the DOM element for streaming
    streamingMessageElement = document.createElement('div');
    streamingMessageElement.classList.add('message', 'assistant');
    streamingMessageElement.innerHTML = '<div class="message-content"></div>';
    messagesContainer.appendChild(streamingMessageElement);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    // Start chunked TTS pipeline if TTS is active
    console.log('[ttsChunker] Pre-stream check: ttsEnabled:', ttsEnabled);
    if (ttsEnabled) ttsChunker.start();

    // Process stream based on provider type
    const streamType = selectedProvider?.type || currentProvider;
    if (streamType === 'ollama' || streamType === 'squatchserve') {
      // Ollama and SquatchServe use NDJSON streaming format
      fullResponse = await processOllamaStream(response);
    } else if (streamType === 'claude') {
      fullResponse = await processClaudeStream(response);
    } else if (streamType === 'grok' || streamType === 'openai' || streamType === 'llamacpp' || streamType === 'vllm') {
      // Grok, OpenAI, Llama.cpp, and vLLM use OpenAI-compatible SSE streaming format
      fullResponse = await processGrokStream(response);
    }

    // Update the assistant message with the complete response
    console.log('[sendMessage] Stream complete, fullResponse length:', fullResponse.length);
    const assistantMessageIndex = conversation.findIndex(msg => msg.id === assistantMessageId);
    if (assistantMessageIndex !== -1) {
      conversation[assistantMessageIndex].content = fullResponse;
      console.log('[sendMessage] Updated conversation at index:', assistantMessageIndex);
    }

    // Cancel any pending animation frame to avoid stale updates
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
    }

    // Clear streaming state BEFORE re-rendering
    streamingMessageElement = null;
    pendingContent = null;
    animationFrameId = null;

    saveConversation();

    // Force re-render to ensure UI matches conversation state
    // This guarantees the response is displayed even if streaming updates failed
    renderMessages();

    console.log('[ttsChunker] Post-stream: ttsEnabled:', ttsEnabled, 'fullResponse length:', fullResponse.length);
    if (ttsEnabled) {
      ttsChunker.flush(fullResponse);
    }

    // Refresh conversation list to show the new/updated conversation
    loadConversations();

  } catch (error) {
    console.error('Error sending message:', error);
    ttsChunker.cancel();
    addMessage('error', `Failed to send message: ${error.message}`);
  } finally {
    hideTypingIndicator();
    // Fix 6: Re-enable send button on all exit paths
    sendBtn.disabled = false;
    sendBtn.classList.remove('btn-disabled');
  }
}

// Show memory context indicator (persistent, clickable, with source breakdown)
function showMemoryIndicator(response) {
  const indicator = document.createElement('div');
  indicator.className = 'memory-indicator-persistent';

  // Parse memory sources from response headers
  let sourcesText = '';
  if (response && response.headers) {
    const sources = response.headers.get('X-Memory-Sources') || 'none';
    if (sources !== 'none') {
      const parts = sources.split(',');
      const labels = parts.map(s => {
        if (s === 'long-term') return 'Long-term memory';
        if (s === 'user-profile') return 'User profile';
        if (s === 'daily-today') return "Today's log";
        if (s === 'daily-yesterday') return "Yesterday's log";
        if (s.endsWith('-conversations')) return `${s.split('-')[0]} past conversations`;
        if (s.endsWith('-clusters')) return `${s.split('-')[0]} memory clusters`;
        return s;
      });
      sourcesText = labels.join(', ');
    }
  }

  indicator.innerHTML = `
    <span>Using memory context</span>
    <div class="memory-indicator-details">${sourcesText ? escapeHtml(sourcesText) : 'Memory files loaded'}</div>
  `;

  indicator.addEventListener('click', () => {
    indicator.classList.toggle('expanded');
  });

  messagesContainer.appendChild(indicator);
}

// Show tools usage indicator (persistent — matches memory indicator behavior)
function showToolsIndicator() {
  const indicator = document.createElement('div');
  indicator.className = 'search-results-indicator';
  indicator.textContent = 'Enhanced with tool results';
  messagesContainer.appendChild(indicator);
}

// Process Ollama streaming response
async function processOllamaStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullResponse = '';
  let buffer = ''; // Buffer for partial lines across chunks

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      // SECURITY FIX: Use stream:true to handle partial UTF-8 sequences
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');

      // Keep the last incomplete line in buffer
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.trim() === '') continue;

        try {
          const data = JSON.parse(line);
          if (data.message && data.message.content) {
            fullResponse += data.message.content;
            updateLastMessage(fullResponse);
          }
        } catch (e) {
          console.error('Error parsing Ollama JSON:', e);
        }
      }
    }

    // Process any remaining buffered content
    if (buffer.trim()) {
      try {
        const data = JSON.parse(buffer);
        if (data.message && data.message.content) {
          fullResponse += data.message.content;
          updateLastMessage(fullResponse);
        }
      } catch (e) {
        console.error('Error parsing final Ollama JSON:', e);
      }
    }
  } finally {
    // SECURITY FIX: Always release the reader lock
    reader.releaseLock();
  }

  return fullResponse;
}

// Process Claude streaming response
async function processClaudeStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullResponse = '';
  let buffer = ''; // Buffer for partial lines across chunks

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      // SECURITY FIX: Use stream:true to handle partial UTF-8 sequences
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');

      // Keep the last incomplete line in buffer
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') continue;

          try {
            const parsed = JSON.parse(data);
            if (parsed.type === 'content_block_delta' && parsed.delta?.text) {
              fullResponse += parsed.delta.text;
              updateLastMessage(fullResponse);
            }
          } catch (e) {
            console.error('Error parsing Claude SSE:', e);
          }
        }
      }
    }

    // Process any remaining buffered content (handles missing trailing newline)
    if (buffer.trim()) {
      if (buffer.startsWith('data: ')) {
        const data = buffer.slice(6);
        if (data !== '[DONE]') {
          try {
            const parsed = JSON.parse(data);
            if (parsed.type === 'content_block_delta' && parsed.delta?.text) {
              fullResponse += parsed.delta.text;
              updateLastMessage(fullResponse);
            }
          } catch (e) {
            console.error('Error parsing final Claude SSE:', e);
          }
        }
      }
    }
  } finally {
    // SECURITY FIX: Always release the reader lock
    reader.releaseLock();
  }

  return fullResponse;
}

// Process Grok streaming response (OpenAI-compatible)
async function processGrokStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullResponse = '';
  let buffer = ''; // Buffer for partial lines across chunks
  let chunkCount = 0;

  console.log('[processGrokStream] Starting stream processing');

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) {
        console.log('[processGrokStream] Stream done, chunks received:', chunkCount);
        break;
      }

      chunkCount++;
      // SECURITY FIX: Use stream:true to handle partial UTF-8 sequences
      buffer += decoder.decode(value, { stream: true });

      // Log first chunk to see the format
      if (chunkCount === 1) {
        console.log('[processGrokStream] First chunk:', buffer.substring(0, 200));
      }

      const lines = buffer.split('\n');

      // Keep the last incomplete line in buffer
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmedLine = line.trim();
        if (!trimmedLine || trimmedLine === 'data: [DONE]') continue;

        let jsonStr = trimmedLine;

        // Handle SSE format (data: {...})
        if (trimmedLine.startsWith('data: ')) {
          jsonStr = trimmedLine.slice(6);
          if (jsonStr === '[DONE]') continue;
        }

        try {
          const parsed = JSON.parse(jsonStr);
          let content = null;

          // OpenAI-compatible streaming format (delta)
          if (parsed.choices?.[0]?.delta?.content) {
            content = parsed.choices[0].delta.content;
          }
          // Non-streaming format (message instead of delta)
          else if (parsed.choices?.[0]?.message?.content) {
            content = parsed.choices[0].message.content;
          }
          // Ollama format
          else if (parsed.message?.content) {
            content = parsed.message.content;
          }
          // Direct content field
          else if (parsed.content) {
            content = parsed.content;
          }
          // Direct text field
          else if (parsed.text) {
            content = parsed.text;
          }
          // Response field (some providers use this)
          else if (parsed.response) {
            content = parsed.response;
          }

          if (content) {
            fullResponse += content;
            updateLastMessage(fullResponse);
          }
        } catch (e) {
          // Not valid JSON, skip this line
        }
      }
    }

    // Process any remaining buffered content (handles missing trailing newline)
    if (buffer.trim()) {
      let jsonStr = buffer.trim();

      // Handle SSE format
      if (jsonStr.startsWith('data: ')) {
        jsonStr = jsonStr.slice(6);
      }

      if (jsonStr && jsonStr !== '[DONE]') {
        try {
          const parsed = JSON.parse(jsonStr);
          let content = null;

          if (parsed.choices?.[0]?.delta?.content) {
            content = parsed.choices[0].delta.content;
          } else if (parsed.choices?.[0]?.message?.content) {
            content = parsed.choices[0].message.content;
          } else if (parsed.message?.content) {
            content = parsed.message.content;
          } else if (parsed.content) {
            content = parsed.content;
          } else if (parsed.text) {
            content = parsed.text;
          } else if (parsed.response) {
            content = parsed.response;
          }

          if (content) {
            fullResponse += content;
            updateLastMessage(fullResponse);
          }
        } catch (e) {
          // Not valid JSON, ignore
        }
      }
    }
  } finally {
    // SECURITY FIX: Always release the reader lock
    reader.releaseLock();
  }

  console.log('[processGrokStream] Final response length:', fullResponse.length);
  if (fullResponse.length === 0) {
    console.log('[processGrokStream] WARNING: No content extracted! Remaining buffer:', buffer);
  }

  return fullResponse;
}

// Handle key down events (Enter to send, Shift+Enter for new line)
function handleKeyDown(event) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
}

// Auto-resize textarea
function autoResizeInput() {
  messageInput.style.height = 'auto';
  messageInput.style.height = (messageInput.scrollHeight) + 'px';
}

// Add message to conversation
function addMessage(role, content) {
  conversation.push({
    role,
    content,
    id: Date.now()
  });
  
  saveConversation();
  renderMessages();
}

// Update the last message (for streaming responses)
function updateLastMessage(content) {
  if (ttsEnabled) {
    console.log('[ttsChunker] updateLastMessage feeding, content length:', content.length, 'active:', ttsChunker.active);
    ttsChunker.feed(content);
  }
  if (streamingMessageElement) {
    pendingContent = content;
    
    // Only update once per animation frame (usually 60fps)
    if (!animationFrameId) {
      animationFrameId = requestAnimationFrame(() => {
        if (pendingContent && streamingMessageElement) {
          const contentElement = streamingMessageElement.querySelector('.message-content');
          contentElement.innerHTML = formatMessageContent(pendingContent);
          chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        animationFrameId = null;
      });
    }
  }
}

// Render messages to the chat container
function renderMessages() {
  messagesContainer.innerHTML = '';
  
  conversation.forEach((msg, index) => {
    const messageElement = document.createElement('div');
    messageElement.classList.add('message', msg.role);
    
    if (msg.role === 'user') {
      messageElement.innerHTML = `<div class="message-content">${escapeHtml(msg.content)}</div>`;
    } else if (msg.role === 'assistant') {
      messageElement.innerHTML = `<div class="message-content">${formatMessageContent(msg.content)}</div>`;
    } else if (msg.role === 'system') {
      messageElement.innerHTML = `<div class="message-content system-message">${escapeHtml(msg.content)}</div>`;
    } else if (msg.role === 'error') {
      messageElement.innerHTML = `<div class="message-content error-message">${escapeHtml(msg.content)}</div>`;
    }
    
    messagesContainer.appendChild(messageElement);
  });
  
  // Scroll to bottom
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Format message content (handle markdown-like formatting)
function formatMessageContent(content) {
  // SECURITY: Escape HTML first to prevent XSS, then apply markdown formatting
  const escaped = escapeHtml(content);
  return escaped
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`(.*?)`/g, '<code>$1</code>')
    .replace(/\n/g, '<br>');
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Show typing indicator
function showTypingIndicator() {
  isTyping = true;
  typingIndicator.style.display = 'flex';
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Hide typing indicator
function hideTypingIndicator() {
  isTyping = false;
  typingIndicator.style.display = 'none';
}

// Start a new chat
function newChat() {
  startNewConversation();
}

// TTS Toggle
function toggleTTS() {
  ttsEnabled = !ttsEnabled;
  speakerToggle.textContent = ttsEnabled ? '🔊' : '🔇';
  speakerToggle.classList.toggle('enabled', ttsEnabled);
  
  // Unlock audio on first interaction
  if (!audioUnlocked) {
    const silence = new Audio("data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=");
    silence.play().then(() => audioUnlocked = true).catch(() => {});
  }
}

// Super Search Toggle
function toggleSuperSearch() {
  superSearch = !superSearch;
  superSearchBtn.classList.toggle('active', superSearch);
}


// Recording functions
async function startRecording() {
  // Unlock audio on mic interaction
  if (!audioUnlocked) {
    const silence = new Audio("data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=");
    silence.play().then(() => audioUnlocked = true).catch(() => {});
  }
  
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    
    // Set up audio analysis for silence detection
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    analyser = audioContext.createAnalyser();
    const source = audioContext.createMediaStreamSource(stream);
    source.connect(analyser);
    analyser.fftSize = 512;
    
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];
    
    mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
    mediaRecorder.onstop = sendAudioToWhisper;
    
    mediaRecorder.start();
    isRecording = true;
    micBtn.classList.add('recording');
    
    // Start silence detection if in conversation mode
    if (conversationMode) {
      detectSilence();
    }
  } catch (err) {
    console.error('Mic access error:', err);
    alert('Could not access microphone');
  }
}

function stopRecording() {
  if (mediaRecorder && isRecording) {
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(track => track.stop());
    isRecording = false;
    micBtn.classList.remove('recording');
    if (silenceTimer) {
      clearTimeout(silenceTimer);
      silenceTimer = null;
    }
  }
}

async function sendAudioToWhisper() {
  const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });

  try {
    const response = await fetch(STT_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'audio/webm' },
      body: audioBlob
    });
    const data = await response.json();
    if (data.text) {
      messageInput.value = data.text;
      sendMessage();
    }
  } catch (err) {
    console.error('STT error:', err);
  }
}

async function speakText(text) {
  if (!ttsEnabled) return;

  // Clean markdown formatting
  text = text
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*(.*?)\*/g, '$1')
    .replace(/`(.*?)`/g, '$1')
    .replace(/#{1,6}\s?/g, '')
    .replace(/\n/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  if (!text) return;

  try {
    const response = await fetch(TTS_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text })
    });

    if (!response.ok) {
      console.error('TTS request failed:', response.status);
      return;
    }

    const audioBlob = await response.blob();
    if (audioBlob.size === 0) {
      console.error('TTS returned empty audio');
      return;
    }

    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio();
    audio.src = audioUrl;

    audio.onended = () => {
      URL.revokeObjectURL(audioUrl);
      if (conversationMode && !isRecording) {
        startRecording();
      }
    };

    audio.onerror = (e) => {
      console.error('Audio playback error:', e);
      URL.revokeObjectURL(audioUrl);
    };

    await audio.play().catch(e => {
      console.error('Audio play failed:', e);
      const playBtn = document.createElement('button');
      playBtn.textContent = '▶️ Play Response';
      playBtn.className = 'play-response-btn';
      playBtn.onclick = () => {
        audio.play();
        playBtn.remove();
        audioUnlocked = true;
      };
      messagesContainer.appendChild(playBtn);
    });
  } catch (err) {
    console.error('TTS error:', err);
  }
}

// Silence detection for conversation mode
function detectSilence() {
  // Wait 2 seconds before starting silence detection
  setTimeout(() => {
  const bufferLength = analyser.fftSize;
  const dataArray = new Uint8Array(bufferLength);
  let silenceStart = null;
  const silenceThreshold = 5;  // Adjust if needed
  const silenceDuration = 4000; // 4 seconds of silence
  
  function checkAudio() {
    if (!isRecording || !conversationMode) return;
    
    analyser.getByteTimeDomainData(dataArray);
    
    // Calculate volume
    let sum = 0;
    for (let i = 0; i < bufferLength; i++) {
      const val = (dataArray[i] - 128) / 128;
      sum += val * val;
    }
    const volume = Math.sqrt(sum / bufferLength) * 100;
    
    if (volume < silenceThreshold) {
      if (!silenceStart) silenceStart = Date.now();
      else if (Date.now() - silenceStart > silenceDuration) {
        stopRecording();
        return;
      }
    } else {
      silenceStart = null;
    }
    
    requestAnimationFrame(checkAudio);
  }
  
  checkAudio();
  }, 2000);
}

// Toggle conversation mode
function toggleConversationMode() {
  conversationMode = !conversationMode;
  convoModeBtn.classList.toggle('active', conversationMode);
  convoModeBtn.textContent = conversationMode ? '🗣️' : '💬';

  // Also enable TTS when entering conversation mode
  if (conversationMode) {
    ttsEnabled = true;
    speakerToggle.textContent = '🔊';
    speakerToggle.classList.add('enabled');

    // Unlock audio
    const silence = new Audio("data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=");
    silence.play().then(() => audioUnlocked = true).catch(() => {});

    // Start listening
    startRecording();
  } else {
    stopRecording();
  }
}

// Settings Modal Functions

// Track which tabs have been loaded this session to avoid re-rendering and losing edits
const settingsTabsLoaded = new Set();

function openSettings() {
  // Clear loaded state so tabs re-fetch fresh data each time the modal opens
  settingsTabsLoaded.clear();
  settingsModal.style.display = 'flex';
  const activeTab = document.querySelector('.settings-nav-item.active');
  const tabName = activeTab ? activeTab.dataset.settingsTab : 'chat';
  switchSettingsTab(tabName);
}

function closeSettings() {
  settingsModal.style.display = 'none';
}

// loadSettings is called at init — just a no-op now since tabs load on demand
function loadSettings() {
  // Values are loaded per-tab when each tab is opened
}

// Fix 8: Escape key — close settings modal or memory panel (single listener)
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    if (settingsModal.style.display !== 'none') {
      closeSettings();
    } else if (memoryPanel?.classList.contains('open')) {
      closeMemoryPanel();
    }
  }
});

function switchSettingsTab(name) {
  document.querySelectorAll('.settings-nav-item').forEach(t => t.classList.remove('active'));
  document.querySelector(`.settings-nav-item[data-settings-tab="${name}"]`)?.classList.add('active');
  document.querySelectorAll('.settings-tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById(`settingsTab${name.charAt(0).toUpperCase() + name.slice(1)}`)?.classList.add('active');

  // Only load tab content if not already loaded this modal session
  if (!settingsTabsLoaded.has(name)) {
    settingsTabsLoaded.add(name);
    if (name === 'chat') loadSettingsChatTab();
    else if (name === 'brain') loadSettingsBrainTab();
    else if (name === 'voice') loadSettingsVoiceTab();
    else if (name === 'tools') loadSettingsToolsTab();
    else if (name === 'about') loadSettingsAboutTab();
  }
}

let chatTabLoadGeneration = 0;
let toolsTabLoadGeneration = 0;

async function loadSettingsChatTab() {
  const container = document.getElementById('settingsTabChat');
  if (!container) return;

  const generation = ++chatTabLoadGeneration;

  const claudeKey = localStorage.getItem('claudeApiKey') || '';
  const openaiKey = localStorage.getItem('openaiApiKey') || '';
  const grokKey = localStorage.getItem('grokApiKey') || '';
  const squatchserveHost = localStorage.getItem('squatchserveHost') || '';

  // Load instances from config via providers endpoint
  let instances = { ollama: [], vllm: [], llamacpp: [] };
  try {
    const res = await fetch('/api/providers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        hasClaudeKey: !!claudeKey,
        hasGrokKey: !!localStorage.getItem('grokApiKey'),
        hasOpenAIKey: !!localStorage.getItem('openaiApiKey')
      })
    });
    if (res.ok) {
      const data = await res.json();
      instances = data.instances || instances;
    }
  } catch (e) {
    console.error('[Settings] Failed to load instances:', e);
  }

  if (generation !== chatTabLoadGeneration) return;

  container.innerHTML = '';

  // Instance managers for each provider type
  const providerTypes = [
    { key: 'ollama', label: 'Ollama', hasModel: false },
    { key: 'vllm', label: 'vLLM', hasModel: true },
    { key: 'llamacpp', label: 'Llama.cpp', hasModel: true }
  ];

  for (const pt of providerTypes) {
    const section = document.createElement('div');
    section.className = 'settings-section';

    const h3 = document.createElement('h3');
    h3.textContent = `${pt.label} Instances`;
    section.appendChild(h3);

    const hint = document.createElement('p');
    hint.className = 'settings-hint';
    hint.textContent = pt.hasModel
      ? `Named ${pt.label} server connections with model name`
      : `Named ${pt.label} server connections (models fetched live)`;
    section.appendChild(hint);

    // Instance list
    const listDiv = document.createElement('div');
    listDiv.className = 'instance-list';
    listDiv.id = `instance-list-${pt.key}`;

    const currentInstances = instances[pt.key] || [];
    for (const inst of currentInstances) {
      listDiv.appendChild(createInstanceItem(pt.key, inst, pt.hasModel));
    }

    if (currentInstances.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'instance-empty';
      empty.textContent = 'No instances configured';
      listDiv.appendChild(empty);
    }

    section.appendChild(listDiv);

    // Add form
    const addForm = document.createElement('div');
    addForm.className = 'instance-add-form';
    addForm.innerHTML = `
      <input type="text" placeholder="Name" class="instance-input instance-name-input" data-provider="${pt.key}">
      <input type="text" placeholder="Host URL" class="instance-input instance-host-input" data-provider="${pt.key}">
      ${pt.hasModel ? `<input type="text" placeholder="Model" class="instance-input instance-model-input" data-provider="${pt.key}">` : ''}
      <button class="instance-add-btn" data-provider="${pt.key}" data-has-model="${pt.hasModel}">Add</button>
    `;
    section.appendChild(addForm);

    // Wire up Add button
    addForm.querySelector('.instance-add-btn').addEventListener('click', (e) => {
      handleAddInstance(e, pt.key, pt.hasModel);
    });

    container.appendChild(section);
  }

  // SquatchServe host (single instance, not converted to instance manager)
  const sqSection = document.createElement('div');
  sqSection.className = 'settings-section';
  sqSection.innerHTML = `
    <h3>SquatchServe</h3>
    <p class="settings-hint">Single SquatchServe server connection</p>
    <div class="setting-item">
      <label for="settings-squatchserveHost">Host</label>
      <input type="text" id="settings-squatchserveHost" class="api-key-input" placeholder="http://localhost:8111" value="${escapeHtml(squatchserveHost)}">
    </div>
  `;
  container.appendChild(sqSection);

  // API Keys section
  const keysSection = document.createElement('div');
  keysSection.className = 'settings-section';
  keysSection.innerHTML = `
    <h3>API Keys</h3>
    <p class="settings-hint">API keys are stored locally in your browser and never sent to our servers</p>
    <div class="setting-item">
      <label for="settings-claudeApiKey">Claude API Key</label>
      <div class="api-key-input-wrapper">
        <input type="password" id="settings-claudeApiKey" class="api-key-input" placeholder="sk-ant-..." value="${escapeHtml(claudeKey)}">
        <button class="toggle-visibility-btn" data-target="settings-claudeApiKey">👁️</button>
      </div>
    </div>
    <div class="setting-item">
      <label for="settings-openaiApiKey">OpenAI API Key</label>
      <div class="api-key-input-wrapper">
        <input type="password" id="settings-openaiApiKey" class="api-key-input" placeholder="sk-..." value="${escapeHtml(openaiKey)}">
        <button class="toggle-visibility-btn" data-target="settings-openaiApiKey">👁️</button>
      </div>
    </div>
    <div class="setting-item">
      <label for="settings-grokApiKey">Grok API Key</label>
      <div class="api-key-input-wrapper">
        <input type="password" id="settings-grokApiKey" class="api-key-input" placeholder="xai-..." value="${escapeHtml(grokKey)}">
        <button class="toggle-visibility-btn" data-target="settings-grokApiKey">👁️</button>
      </div>
    </div>
  `;
  container.appendChild(keysSection);

  // Wire up visibility toggles
  container.querySelectorAll('.toggle-visibility-btn').forEach(btn => {
    btn.addEventListener('click', togglePasswordVisibility);
  });
}

function createInstanceItem(providerType, inst, hasModel) {
  const item = document.createElement('div');
  item.className = 'instance-item';

  const info = document.createElement('div');
  info.className = 'instance-info';

  const nameSpan = document.createElement('span');
  nameSpan.className = 'instance-name';
  nameSpan.textContent = inst.name;
  info.appendChild(nameSpan);

  const hostSpan = document.createElement('span');
  hostSpan.className = 'instance-host';
  hostSpan.textContent = inst.host;
  info.appendChild(hostSpan);

  if (hasModel && inst.model) {
    const modelSpan = document.createElement('span');
    modelSpan.className = 'instance-model';
    modelSpan.textContent = inst.model;
    info.appendChild(modelSpan);
  }

  item.appendChild(info);

  const deleteBtn = document.createElement('button');
  deleteBtn.className = 'instance-delete-btn';
  deleteBtn.textContent = 'Delete';
  deleteBtn.addEventListener('click', () => {
    handleDeleteInstance(providerType, inst.name);
  });
  item.appendChild(deleteBtn);

  return item;
}

async function handleAddInstance(event, providerType, hasModel) {
  const form = event.target.closest('.instance-add-form');
  const nameInput = form.querySelector('.instance-name-input');
  const hostInput = form.querySelector('.instance-host-input');
  const modelInput = hasModel ? form.querySelector('.instance-model-input') : null;

  const name = nameInput.value.trim();
  const host = hostInput.value.trim();
  const model = modelInput ? modelInput.value.trim() : undefined;

  if (!name || !host) {
    alert('Name and Host are required.');
    return;
  }
  if (hasModel && !model) {
    alert('Model is required for this provider type.');
    return;
  }

  const newInst = { name, host };
  if (hasModel) newInst.model = model;

  try {
    const configRes = await fetch('/api/config');
    if (!configRes.ok) throw new Error('Failed to load config');
    const config = await configRes.json();

    const currentInstances = Array.isArray(config.providers?.[providerType]) ? config.providers[providerType] : [];

    if (currentInstances.some(i => i.name === name)) {
      alert(`An instance named "${name}" already exists for this provider.`);
      return;
    }

    currentInstances.push(newInst);
    const partial = { providers: { [providerType]: currentInstances } };

    const saveRes = await fetch('/api/config', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(partial)
    });
    if (!saveRes.ok) throw new Error('Failed to save');

    // Clear inputs
    nameInput.value = '';
    hostInput.value = '';
    if (modelInput) modelInput.value = '';

    // Clear tab cache and reload
    settingsTabsLoaded.delete('chat');
    loadSettingsChatTab();
    loadProviders();
  } catch (error) {
    console.error('[Settings] Error adding instance:', error);
    alert('Failed to add instance: ' + error.message);
  }
}

async function handleDeleteInstance(providerType, instanceName) {
  if (!confirm(`Delete instance "${instanceName}"?`)) return;

  try {
    const configRes = await fetch('/api/config');
    if (!configRes.ok) throw new Error('Failed to load config');
    const config = await configRes.json();

    const currentInstances = Array.isArray(config.providers?.[providerType]) ? config.providers[providerType] : [];
    const filtered = currentInstances.filter(i => i.name !== instanceName);

    const partial = { providers: { [providerType]: filtered } };

    const saveRes = await fetch('/api/config', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(partial)
    });
    if (!saveRes.ok) throw new Error('Failed to save');

    // Clear tab cache and reload
    settingsTabsLoaded.delete('chat');
    loadSettingsChatTab();
    loadProviders();
  } catch (error) {
    console.error('[Settings] Error deleting instance:', error);
    alert('Failed to delete instance: ' + error.message);
  }
}

async function loadSettingsBrainTab() {
  const container = document.getElementById('settingsTabBrain');
  if (!container) return;
  container.innerHTML = '<div class="config-loading">Loading configuration...</div>';

  try {
    // Load config and provider instances in parallel
    const [configRes, providersRes] = await Promise.all([
      fetch('/api/config'),
      fetch('/api/providers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          hasClaudeKey: !!localStorage.getItem('claudeApiKey'),
          hasGrokKey: !!localStorage.getItem('grokApiKey'),
          hasOpenAIKey: !!localStorage.getItem('openaiApiKey')
        })
      })
    ]);

    if (!configRes.ok || !providersRes.ok) throw new Error('Failed to load config');
    const config = await configRes.json();
    const providersData = await providersRes.json();

    container.innerHTML = '';

    // Memory Thresholds
    container.appendChild(createConfigSection('Memory Thresholds', [
      { key: 'memory.similarityThreshold', label: 'Similarity Threshold', type: 'number', value: config.memory?.similarityThreshold, step: '0.05' },
      { key: 'memory.clusterLinkThreshold', label: 'Cluster Link Threshold', type: 'number', value: config.memory?.clusterLinkThreshold, step: '0.05' },
      { key: 'memory.maxFactsPerCluster', label: 'Max Facts Per Cluster (triggers audit)', type: 'number', value: config.memory?.maxFactsPerCluster, step: '1' },
      { key: 'memory.dailyLogRetentionDays', label: 'Daily Log Retention (days)', type: 'number', value: config.memory?.dailyLogRetentionDays, step: '1' },
      { key: 'memory.hybridSearchWeights.vector', label: 'Vector Weight', type: 'number', value: config.memory?.hybridSearchWeights?.vector, step: '0.1' },
      { key: 'memory.hybridSearchWeights.bm25', label: 'BM25 Weight', type: 'number', value: config.memory?.hybridSearchWeights?.bm25, step: '0.1' }
    ]));

    // Build instance options for select dropdowns (only local instance-based providers)
    const instanceOptions = [];
    const instances = providersData.instances || {};
    for (const providerType of ['ollama', 'vllm', 'llamacpp']) {
      const typeLabel = providerType === 'ollama' ? 'Ollama' : providerType === 'vllm' ? 'vLLM' : 'Llama.cpp';
      for (const inst of (instances[providerType] || [])) {
        instanceOptions.push({
          value: `${providerType}:${inst.name}`,
          label: `${typeLabel} — ${inst.name}`
        });
      }
    }

    // Role assignment sections
    const roles = [
      { key: 'chat', label: 'Chat Default' },
      { key: 'extraction', label: 'Fact Extraction' },
      { key: 'heartbeat', label: 'Heartbeat' },
      { key: 'embedding', label: 'Embedding' }
    ];

    for (const role of roles) {
      const roleConfig = config.models?.[role.key] || {};
      const currentValue = roleConfig.provider && roleConfig.instance
        ? `${roleConfig.provider}:${roleConfig.instance}`
        : roleConfig.provider ? `${roleConfig.provider}:Local` : '';

      const section = document.createElement('div');
      section.className = 'config-section';
      const h3 = document.createElement('h3');
      h3.textContent = role.label;
      section.appendChild(h3);

      // Instance selector
      const instanceItem = document.createElement('div');
      instanceItem.className = 'config-item';
      const instanceLabel = document.createElement('label');
      instanceLabel.textContent = 'Instance';
      const instanceLabelId = `brain-${role.key}-instance`;
      instanceLabel.setAttribute('for', instanceLabelId);
      instanceItem.appendChild(instanceLabel);

      const instanceSelect = document.createElement('select');
      instanceSelect.id = instanceLabelId;
      instanceSelect.dataset.brainRole = role.key;
      instanceSelect.dataset.brainField = 'instance';

      for (const opt of instanceOptions) {
        const option = document.createElement('option');
        option.value = opt.value;
        option.textContent = opt.label;
        if (opt.value === currentValue) option.selected = true;
        instanceSelect.appendChild(option);
      }
      instanceItem.appendChild(instanceSelect);
      section.appendChild(instanceItem);

      // Model dropdown (populated dynamically from instance)
      const modelItem = document.createElement('div');
      modelItem.className = 'config-item';
      const modelLabel = document.createElement('label');
      modelLabel.textContent = 'Model';
      const modelLabelId = `brain-${role.key}-model`;
      modelLabel.setAttribute('for', modelLabelId);
      modelItem.appendChild(modelLabel);

      const modelSelect = document.createElement('select');
      modelSelect.id = modelLabelId;
      modelSelect.dataset.brainRole = role.key;
      modelSelect.dataset.brainField = 'model';
      modelItem.appendChild(modelSelect);
      section.appendChild(modelItem);

      // Load models for the current instance selection
      const savedModel = roleConfig.model || '';
      loadBrainRoleModels(instanceSelect.value, modelSelect, savedModel);

      // Refresh models when instance changes
      instanceSelect.addEventListener('change', () => {
        loadBrainRoleModels(instanceSelect.value, modelSelect, '');
      });

      container.appendChild(section);
    }

    // Heartbeat scheduling (separate section)
    container.appendChild(createConfigSection('Heartbeat Schedule', [
      { key: 'heartbeat.enabled', label: 'Enabled', type: 'checkbox', value: config.heartbeat?.enabled },
      { key: 'heartbeat.intervalHours', label: 'Interval (hours)', type: 'number', value: config.heartbeat?.intervalHours, step: '0.5' },
      { key: 'heartbeat.warmupMinutes', label: 'Warmup (minutes)', type: 'number', value: config.heartbeat?.warmupMinutes, step: '1' }
    ]));

    const notice = document.createElement('div');
    notice.className = 'config-notice';
    notice.textContent = 'Heartbeat interval changes require a server restart.';
    container.appendChild(notice);

    // Rebuild Clusters button
    const rebuildSection = document.createElement('div');
    rebuildSection.className = 'config-section';
    const rebuildH3 = document.createElement('h3');
    rebuildH3.textContent = 'Cluster Maintenance';
    rebuildSection.appendChild(rebuildH3);

    const rebuildBtn = document.createElement('button');
    rebuildBtn.className = 'config-btn';
    rebuildBtn.textContent = 'Rebuild Clusters';
    rebuildBtn.title = 'Run a full intelligent audit and reorganization of all memory clusters';
    rebuildBtn.addEventListener('click', async () => {
      if (!confirm('This will run a full LLM-driven audit of all clusters. This may take several minutes. Continue?')) return;
      rebuildBtn.disabled = true;
      rebuildBtn.textContent = 'Rebuilding...';
      try {
        const res = await fetch('/api/memory/rebuild', { method: 'POST' });
        const result = await res.json();
        if (result.skipped) {
          alert('A heartbeat cycle is already running. Try again later.');
        } else if (result.error) {
          alert('Rebuild failed: ' + result.error);
        } else {
          alert(`Rebuild complete!\n\nClusters audited: ${result.report?.clustersAudited || 0}\nClusters split: ${result.report?.clustersSplit || 0}\nLinks updated: ${result.report?.linksUpdated || 0}\nDuration: ${result.report?.totalDuration || 'unknown'}`);
        }
      } catch (err) {
        alert('Rebuild failed: ' + err.message);
      } finally {
        rebuildBtn.disabled = false;
        rebuildBtn.textContent = 'Rebuild Clusters';
      }
    });
    rebuildSection.appendChild(rebuildBtn);
    container.appendChild(rebuildSection);
  } catch (error) {
    console.error('[Settings] Error loading brain config:', error);
    container.innerHTML = '<div class="config-loading">Failed to load configuration.</div>';
  }
}

async function loadBrainRoleModels(instanceValue, modelSelect, preselectModel) {
  modelSelect.innerHTML = '<option value="">Loading...</option>';

  if (!instanceValue || !instanceValue.includes(':')) {
    modelSelect.innerHTML = '<option value="">Select an instance first</option>';
    return;
  }

  const colonIdx = instanceValue.indexOf(':');
  const providerType = instanceValue.substring(0, colonIdx);
  const instanceName = instanceValue.substring(colonIdx + 1);

  try {
    const res = await fetch('/api/instance/models', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ providerType, instanceName })
    });

    if (!res.ok) throw new Error('Failed to fetch models');
    const data = await res.json();
    const models = data.models || [];

    modelSelect.innerHTML = '';

    if (models.length === 0) {
      modelSelect.innerHTML = '<option value="">No models available</option>';
      return;
    }

    for (const m of models) {
      const option = document.createElement('option');
      option.value = m.id;
      option.textContent = m.name;
      if (m.id === preselectModel) option.selected = true;
      modelSelect.appendChild(option);
    }

    // If no preselect matched, just keep the first option selected
  } catch (error) {
    console.error('[Settings] Error loading models for brain role:', error);
    modelSelect.innerHTML = '<option value="">Failed to load models</option>';
  }
}

async function loadSettingsVoiceTab() {
  const container = document.getElementById('settingsTabVoice');
  if (!container) return;
  container.innerHTML = '<div class="config-loading">Loading voice providers...</div>';

  const STT_TYPES = ['whisper', 'faster-whisper', 'canary', 'parakeet', 'deepgram', 'openai-whisper'];
  const TTS_TYPES = ['kokoro', 'piper', 'chatterbox', 'orpheus', 'qwen3tts', 'elevenlabs', 'openai-tts'];
  const CLOUD_TYPES = new Set(['deepgram', 'openai-whisper', 'elevenlabs', 'openai-tts']);

  let voiceConfig = { stt: { active: '', providers: [] }, tts: { active: '', providers: [] } };
  try {
    const res = await fetch('/api/voice/providers');
    if (res.ok) {
      const data = await res.json();
      if (data.stt) voiceConfig.stt = data.stt;
      if (data.tts) voiceConfig.tts = data.tts;
    }
  } catch (e) {
    console.error('[Settings] Failed to load voice config:', e);
  }

  container.innerHTML = '';

  function buildProviderSection(category, label, types) {
    const catConfig = voiceConfig[category] || { active: '', providers: [] };
    const providers = Array.isArray(catConfig.providers) ? catConfig.providers : [];

    const section = document.createElement('div');
    section.className = 'settings-section';

    const h3 = document.createElement('h3');
    h3.textContent = `${label} Providers`;
    section.appendChild(h3);

    // Provider list
    const listDiv = document.createElement('div');
    listDiv.className = 'instance-list';
    listDiv.id = `voice-list-${category}`;

    for (const p of providers) {
      const item = document.createElement('div');
      item.className = 'instance-item';

      const info = document.createElement('div');
      info.className = 'instance-info';

      const nameSpan = document.createElement('span');
      nameSpan.className = 'instance-name';
      nameSpan.textContent = p.name;
      info.appendChild(nameSpan);

      const typeSpan = document.createElement('span');
      typeSpan.className = 'instance-model';
      typeSpan.textContent = p.type;
      info.appendChild(typeSpan);

      const hostSpan = document.createElement('span');
      hostSpan.className = 'instance-host';
      hostSpan.textContent = CLOUD_TYPES.has(p.type) ? 'cloud' : (p.host || '');
      info.appendChild(hostSpan);

      item.appendChild(info);

      const deleteBtn = document.createElement('button');
      deleteBtn.className = 'instance-delete-btn';
      deleteBtn.textContent = 'Delete';
      deleteBtn.addEventListener('click', () => handleDeleteVoiceProvider(category, p.name));
      item.appendChild(deleteBtn);

      listDiv.appendChild(item);
    }

    if (providers.length === 0) {
      const empty = document.createElement('div');
      empty.className = 'instance-empty';
      empty.textContent = 'No providers configured';
      listDiv.appendChild(empty);
    }

    section.appendChild(listDiv);

    // Add form
    const addForm = document.createElement('div');
    addForm.className = 'instance-add-form';

    const nameInput = document.createElement('input');
    nameInput.type = 'text';
    nameInput.placeholder = 'Name';
    nameInput.className = 'instance-input';
    addForm.appendChild(nameInput);

    const typeSelect = document.createElement('select');
    typeSelect.className = 'instance-input';
    typeSelect.style.minWidth = '120px';
    for (const t of types) {
      const opt = document.createElement('option');
      opt.value = t;
      opt.textContent = t;
      typeSelect.appendChild(opt);
    }
    addForm.appendChild(typeSelect);

    const hostInput = document.createElement('input');
    hostInput.type = 'text';
    hostInput.placeholder = 'Host URL';
    hostInput.className = 'instance-input';
    addForm.appendChild(hostInput);

    const apiKeyInput = document.createElement('input');
    apiKeyInput.type = 'password';
    apiKeyInput.placeholder = 'API Key';
    apiKeyInput.className = 'instance-input';
    apiKeyInput.style.display = 'none';
    addForm.appendChild(apiKeyInput);

    // Toggle host/apikey based on type
    function updateFieldVisibility() {
      const isCloud = CLOUD_TYPES.has(typeSelect.value);
      hostInput.style.display = isCloud ? 'none' : '';
      apiKeyInput.style.display = isCloud ? '' : 'none';
    }
    typeSelect.addEventListener('change', updateFieldVisibility);
    updateFieldVisibility();

    const addBtn = document.createElement('button');
    addBtn.className = 'instance-add-btn';
    addBtn.textContent = 'Add';
    addBtn.addEventListener('click', () => {
      const name = nameInput.value.trim();
      const type = typeSelect.value;
      const isCloud = CLOUD_TYPES.has(type);
      const host = hostInput.value.trim();
      const apiKey = apiKeyInput.value.trim();

      if (!name) { alert('Name is required.'); return; }
      if (!isCloud && !host) { alert('Host URL is required.'); return; }
      if (isCloud && !apiKey) { alert('API Key is required for cloud providers.'); return; }

      if (providers.some(p => p.name === name)) {
        alert(`A provider named "${name}" already exists.`);
        return;
      }

      const newProvider = { name, type };
      if (isCloud) {
        newProvider.api_key = apiKey;
      } else {
        newProvider.host = host;
      }

      providers.push(newProvider);
      saveVoiceConfig(voiceConfig);
    });
    addForm.appendChild(addBtn);

    section.appendChild(addForm);

    // Active provider selector
    const activeItem = document.createElement('div');
    activeItem.className = 'config-item';
    activeItem.style.marginTop = '12px';

    const activeLabel = document.createElement('label');
    activeLabel.textContent = 'Active';
    activeItem.appendChild(activeLabel);

    const activeSelect = document.createElement('select');
    activeSelect.className = 'instance-input';
    activeSelect.style.maxWidth = '220px';
    activeSelect.dataset.voiceCategory = category;
    activeSelect.dataset.voiceField = 'active';

    for (const p of providers) {
      const opt = document.createElement('option');
      opt.value = `${p.type}:${p.name}`;
      const typeLabel = p.type.charAt(0).toUpperCase() + p.type.slice(1);
      opt.textContent = `${typeLabel} — ${p.name}`;
      if (`${p.type}:${p.name}` === catConfig.active) opt.selected = true;
      activeSelect.appendChild(opt);
    }

    if (providers.length === 0) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = 'No providers available';
      activeSelect.appendChild(opt);
    }

    activeItem.appendChild(activeSelect);
    section.appendChild(activeItem);

    return section;
  }

  container.appendChild(buildProviderSection('stt', 'STT', STT_TYPES));
  container.appendChild(buildProviderSection('tts', 'TTS', TTS_TYPES));
}

async function saveVoiceConfig(voiceConfig) {
  try {
    const res = await fetch('/api/voice/providers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(voiceConfig)
    });
    if (!res.ok) {
      const err = await res.json();
      alert('Failed to save: ' + (err.error || 'Unknown error'));
      return;
    }
    // Reload the tab
    settingsTabsLoaded.delete('voice');
    loadSettingsVoiceTab();
  } catch (e) {
    alert('Failed to save: ' + e.message);
  }
}

async function handleDeleteVoiceProvider(category, name) {
  if (!confirm(`Delete voice provider "${name}"?`)) return;

  try {
    const res = await fetch('/api/voice/providers');
    if (!res.ok) throw new Error('Failed to load');
    const voiceConfig = await res.json();

    const cat = voiceConfig[category];
    if (!cat || !Array.isArray(cat.providers)) return;

    cat.providers = cat.providers.filter(p => p.name !== name);

    // If the deleted provider was active, clear or set to first remaining
    const [activeType, ...activeNameParts] = (cat.active || '').split(':');
    const activeName = activeNameParts.join(':');
    if (activeName === name) {
      cat.active = cat.providers.length > 0
        ? `${cat.providers[0].type}:${cat.providers[0].name}`
        : '';
    }

    await saveVoiceConfig(voiceConfig);
  } catch (e) {
    alert('Failed to delete: ' + e.message);
  }
}

async function loadSettingsToolsTab() {
  const container = document.getElementById('settingsTabTools');
  if (!container) return;

  const generation = ++toolsTabLoadGeneration;
  const searxngHost = localStorage.getItem('searxngHost') || '';

  // Fetch current tools config
  let searxngEnabled = false;
  try {
    const resp = await fetch('/api/config');
    if (generation !== toolsTabLoadGeneration) return;
    if (resp.ok) {
      const config = await resp.json();
      searxngEnabled = !!(config.tools && config.tools.searxng && config.tools.searxng.enabled);
    }
  } catch (e) { /* use default */ }
  if (generation !== toolsTabLoadGeneration) return;

  container.innerHTML = `
    <div class="settings-section">
      <h3>Web Search</h3>
      <p class="settings-hint">SearXNG is used for AI-powered web search</p>
      <div class="setting-item toggle-row">
        <label for="settings-searxngEnabled">Enabled</label>
        <label class="toggle-switch">
          <input type="checkbox" id="settings-searxngEnabled" data-config-key="tools.searxng.enabled" ${searxngEnabled ? 'checked' : ''}>
          <span class="toggle-slider"></span>
        </label>
      </div>
      <div class="setting-item">
        <label for="settings-searxngHost">SearXNG Host</label>
        <input type="text" id="settings-searxngHost" class="api-key-input" placeholder="http://192.168.4.97:8888" value="${escapeHtml(searxngHost)}">
      </div>
    </div>
    <div class="settings-section">
      <h3>MCP Connections</h3>
      <p class="settings-hint">Model Context Protocol tool connections — coming soon</p>
    </div>
  `;
}

function loadSettingsAboutTab() {
  const container = document.getElementById('settingsTabAbout');
  if (!container) return;

  container.innerHTML = `
    <div class="settings-section">
      <h3>Squatch Neuro Hub</h3>
      <dl class="about-info">
        <dt>Version</dt>
        <dd>1.0.0</dd>
        <dt>Author</dt>
        <dd>MettaSphere LLC</dd>
        <dt>Description</dt>
        <dd>Neural-linked AI assistant with associative cluster memory and multi-provider support</dd>
        <dt>License</dt>
        <dd>MIT</dd>
      </dl>
    </div>
  `;
}

async function saveSettingsHandler() {
  const statusEl = document.getElementById('settingsStatus');

  // Collect localStorage values (instance management is done via add/delete, not save)
  const localStorageMap = {
    'settings-claudeApiKey': 'claudeApiKey',
    'settings-openaiApiKey': 'openaiApiKey',
    'settings-grokApiKey': 'grokApiKey',
    'settings-squatchserveHost': 'squatchserveHost',
    'settings-searxngHost': 'searxngHost'
  };

  for (const [elId, storageKey] of Object.entries(localStorageMap)) {
    const el = document.getElementById(elId);
    if (el) {
      const val = el.value.trim();
      if (val) localStorage.setItem(storageKey, val);
      else localStorage.removeItem(storageKey);
    }
  }

  // Collect config.json values from all tabs into one partial object
  const partial = {};

  // Standard config-key inputs (Brain memory thresholds, heartbeat schedule)
  document.querySelectorAll('#settingsTabBrain [data-config-key], #settingsTabChat [data-config-key], #settingsTabTools [data-config-key]').forEach(input => {
    const keys = input.dataset.configKey.split('.');
    let obj = partial;
    for (let i = 0; i < keys.length - 1; i++) {
      if (!obj[keys[i]]) obj[keys[i]] = {};
      obj = obj[keys[i]];
    }
    const lastKey = keys[keys.length - 1];
    if (input.type === 'checkbox') obj[lastKey] = input.checked;
    else if (input.type === 'number') {
      const num = parseFloat(input.value);
      if (!isNaN(num)) obj[lastKey] = num;
    } else obj[lastKey] = input.value;
  });

  // Brain role assignments (instance + model)
  const brainSelects = document.querySelectorAll('#settingsTabBrain [data-brain-role][data-brain-field="instance"]');
  const brainModels = document.querySelectorAll('#settingsTabBrain [data-brain-role][data-brain-field="model"]');

  brainSelects.forEach(select => {
    const role = select.dataset.brainRole;
    const value = select.value;
    const colonIdx = value.indexOf(':');
    const provider = colonIdx >= 0 ? value.substring(0, colonIdx) : value;
    const instance = colonIdx >= 0 ? value.substring(colonIdx + 1) : 'Local';
    if (!partial.models) partial.models = {};
    if (!partial.models[role]) partial.models[role] = {};
    partial.models[role].provider = provider;
    partial.models[role].instance = instance;
  });

  brainModels.forEach(input => {
    const role = input.dataset.brainRole;
    if (!partial.models) partial.models = {};
    if (!partial.models[role]) partial.models[role] = {};
    partial.models[role].model = input.value;
  });

  // Voice active selections
  document.querySelectorAll('#settingsTabVoice [data-voice-category][data-voice-field="active"]').forEach(select => {
    const category = select.dataset.voiceCategory;
    if (!partial.voice) partial.voice = {};
    if (!partial.voice[category]) partial.voice[category] = {};
    partial.voice[category].active = select.value;
  });

  // Save to config.json if there is anything to save
  const hasConfigChanges = Object.keys(partial).length > 0;
  if (hasConfigChanges) {
    try {
      const res = await fetch('/api/config', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(partial)
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error || 'Failed to save config');
      }
    } catch (error) {
      console.error('[Settings] Error saving config:', error);
      if (statusEl) {
        statusEl.className = 'config-status error';
        statusEl.textContent = 'Failed to save: ' + error.message;
        setTimeout(() => { statusEl.textContent = ''; statusEl.className = 'config-status'; }, 5000);
      }
      return;
    }
  }

  if (statusEl) {
    statusEl.className = 'config-status success';
    statusEl.textContent = 'Settings saved.';
    setTimeout(() => { statusEl.textContent = ''; statusEl.className = 'config-status'; }, 3000);
  }

  await loadProviders();
}

function togglePasswordVisibility(event) {
  const button = event.currentTarget;
  const targetId = button.getAttribute('data-target');
  const input = document.getElementById(targetId);
  if (!input) return;

  if (input.type === 'password') {
    input.type = 'text';
    button.textContent = '🙈';
  } else {
    input.type = 'password';
    button.textContent = '👁️';
  }
}

// ============ Conversation History Functions ============

// DOM elements for sidebar
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebar-toggle');
const sidebarNewChatBtn = document.getElementById('sidebar-new-chat-btn');
const conversationList = document.getElementById('conversation-list');
const sidebarOverlay = document.getElementById('sidebar-overlay');
const mainContent = document.querySelector('.main-content');

// Set up sidebar event listeners
function setupSidebarListeners() {
  if (sidebarToggle) {
    sidebarToggle.addEventListener('click', toggleSidebar);
  }
  if (sidebarNewChatBtn) {
    sidebarNewChatBtn.addEventListener('click', startNewConversation);
  }
  if (sidebarOverlay) {
    sidebarOverlay.addEventListener('click', closeSidebarOnMobile);
  }

  // Handle window resize
  window.addEventListener('resize', checkMobileView);
}

// Check if we're on mobile and adjust sidebar
function checkMobileView() {
  if (window.innerWidth <= 768) {
    sidebar?.classList.add('collapsed');
    mainContent?.classList.add('sidebar-collapsed');
  }
}

// Toggle sidebar visibility
function toggleSidebar() {
  if (window.innerWidth <= 768) {
    sidebar?.classList.toggle('open');
    sidebarOverlay?.classList.toggle('active');
  } else {
    sidebar?.classList.toggle('collapsed');
    mainContent?.classList.toggle('sidebar-collapsed');
    sidebarCollapsed = !sidebarCollapsed;
    localStorage.setItem('sidebarCollapsed', sidebarCollapsed);
  }
}

// Close sidebar on mobile when clicking overlay
function closeSidebarOnMobile() {
  sidebar?.classList.remove('open');
  sidebarOverlay?.classList.remove('active');
}

// Load all conversations from the server
async function loadConversations() {
  try {
    const response = await fetch('/api/conversations');
    if (!response.ok) {
      throw new Error('Failed to load conversations');
    }
    conversations = await response.json();
    renderConversationList();

    // If no current conversation, show welcome message
    if (!currentConversationId) {
      loadConversation(); // Load from session storage or show welcome
    }
  } catch (error) {
    console.error('Error loading conversations:', error);
    // Fall back to local session storage
    loadConversation();
  }
}

// Render the conversation list in the sidebar
function renderConversationList() {
  if (!conversationList) return;

  if (conversations.length === 0) {
    conversationList.innerHTML = '<div class="conversation-list-empty">No conversations yet.<br>Start a new chat!</div>';
    return;
  }

  conversationList.innerHTML = conversations.map(conv => {
    const isActive = conv.id === currentConversationId;
    const isSnh = conv.initiated_by === 'snh';
    const title = conv.title || 'New Conversation';
    const preview = conv.preview ? conv.preview.substring(0, 40) + '...' : '';
    const timestamp = formatRelativeTime(conv.updated_at);
    const model = conv.model_used ? conv.model_used.split(':')[0] : '';

    return `
      <div class="conversation-item ${isActive ? 'active' : ''} ${isSnh ? 'snh-initiated' : ''}" data-id="${conv.id}">
        <div class="conversation-title">${isSnh ? '<span class="snh-badge" title="SNH reached out">✦ SNH</span> ' : ''}${escapeHtml(title)}</div>
        <div class="conversation-meta">
          <span class="conversation-timestamp">${timestamp}</span>
          ${model ? `<span class="conversation-model">${escapeHtml(model)}</span>` : ''}
        </div>
        <div class="conversation-actions">
          <button class="conversation-action-btn rename" title="Rename" data-id="${conv.id}">✏️</button>
          <button class="conversation-action-btn delete" title="Delete" data-id="${conv.id}">🗑️</button>
        </div>
      </div>
    `;
  }).join('');

  // Add click handlers
  conversationList.querySelectorAll('.conversation-item').forEach(item => {
    item.addEventListener('click', (e) => {
      if (!e.target.closest('.conversation-action-btn')) {
        loadConversationById(item.dataset.id);
      }
    });
  });

  // Add action button handlers
  conversationList.querySelectorAll('.conversation-action-btn.rename').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      renameConversation(btn.dataset.id);
    });
  });

  conversationList.querySelectorAll('.conversation-action-btn.delete').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      deleteConversation(btn.dataset.id);
    });
  });
}

// Format relative time (e.g., "2 hours ago", "Yesterday")
function formatRelativeTime(dateString) {
  if (!dateString) return '';

  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7) return `${diffDays}d ago`;

  return date.toLocaleDateString();
}

// Load a specific conversation by ID
async function loadConversationById(id) {
  try {
    const response = await fetch(`/api/conversations/${id}`);
    if (!response.ok) {
      throw new Error('Failed to load conversation');
    }

    const data = await response.json();
    currentConversationId = id;

    // Convert messages to our format
    conversation = data.messages.map(msg => ({
      role: msg.role,
      content: msg.content,
      id: msg.id
    }));

    // Update model if stored
    if (data.model_used && modelSelect) {
      const modelName = data.model_used;
      // Try to find and select the model
      const option = Array.from(modelSelect.options).find(opt => opt.value === modelName);
      if (option) {
        modelSelect.value = modelName;
        currentModel = modelName;
      }
    }

    renderMessages();
    renderConversationList(); // Update active state
    closeSidebarOnMobile();

  } catch (error) {
    console.error('Error loading conversation:', error);
    addMessage('error', 'Failed to load conversation');
  }
}

// Start a new conversation
function startNewConversation() {
  currentConversationId = null;
  conversation = [];
  lastAssistantMessageId = null;
  sessionStorage.removeItem('ollamaChatConversation');
  renderMessages();
  renderConversationList();
  messageInput.value = '';
  autoResizeInput();
  addMessage('system', 'Welcome! Start a new conversation.');
  closeSidebarOnMobile();
}

// Rename a conversation
async function renameConversation(id) {
  const conv = conversations.find(c => c.id === id);
  const currentTitle = conv?.title || 'New Conversation';
  const newTitle = prompt('Enter new title:', currentTitle);

  if (newTitle && newTitle.trim() !== currentTitle) {
    try {
      const response = await fetch(`/api/conversations/${id}/title`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: newTitle.trim() })
      });

      if (!response.ok) {
        throw new Error('Failed to rename conversation');
      }

      // Update local state and re-render
      const idx = conversations.findIndex(c => c.id === id);
      if (idx !== -1) {
        conversations[idx].title = newTitle.trim();
        renderConversationList();
      }
    } catch (error) {
      console.error('Error renaming conversation:', error);
      alert('Failed to rename conversation');
    }
  }
}

// Delete a conversation
async function deleteConversation(id) {
  if (!confirm('Are you sure you want to delete this conversation?')) {
    return;
  }

  try {
    const response = await fetch(`/api/conversations/${id}`, {
      method: 'DELETE'
    });

    if (!response.ok) {
      throw new Error('Failed to delete conversation');
    }

    // Remove from local state
    conversations = conversations.filter(c => c.id !== id);

    // If deleted current conversation, start new one
    if (id === currentConversationId) {
      startNewConversation();
    } else {
      renderConversationList();
    }
  } catch (error) {
    console.error('Error deleting conversation:', error);
    alert('Failed to delete conversation');
  }
}

// ============ SquatchServe Model Status Functions ============

// Fetch SquatchServe status (loaded models)
async function fetchSquatchserveStatus() {
  if (currentProvider !== 'squatchserve') {
    return;
  }

  try {
    const squatchserveHost = localStorage.getItem('squatchserveHost') || '';
    const url = squatchserveHost
      ? `/api/squatchserve/ps?host=${encodeURIComponent(squatchserveHost)}`
      : '/api/squatchserve/ps';

    const response = await fetch(url);
    if (!response.ok) {
      throw new Error('Failed to fetch status');
    }

    const data = await response.json();
    updateModelStatusDisplay(data);
  } catch (error) {
    console.error('Error fetching SquatchServe status:', error);
    // Show error state but don't spam console
    if (modelStatusBar) {
      modelStatusText.textContent = 'SquatchServe unavailable';
      modelStatusText.classList.remove('loaded');
      unloadModelBtn.style.display = 'none';
    }
  }
}

// Update the model status display
function updateModelStatusDisplay(data) {
  if (!modelStatusBar) return;

  const loadedModels = data.models || [];
  const gpu = data.gpu || {};

  if (loadedModels.length > 0) {
    const model = loadedModels[0]; // Show first loaded model
    loadedSquatchserveModel = model.name;

    // Format VRAM info if available
    let vramInfo = '';
    if (model.vram && model.vram.used_gb) {
      vramInfo = ` (${model.vram.used_gb.toFixed(1)}GB VRAM)`;
    } else if (gpu.used_gb) {
      vramInfo = ` (${gpu.used_gb.toFixed(1)}/${gpu.total_gb.toFixed(1)}GB VRAM)`;
    }

    modelStatusText.textContent = `Loaded: ${model.name}${vramInfo}`;
    modelStatusText.classList.add('loaded');
    unloadModelBtn.style.display = 'inline-block';
  } else {
    loadedSquatchserveModel = null;
    modelStatusText.textContent = 'No model loaded';
    modelStatusText.classList.remove('loaded');
    unloadModelBtn.style.display = 'none';
  }
}

// Unload the currently loaded model
async function unloadSquatchserveModel() {
  if (!loadedSquatchserveModel) {
    return;
  }

  const modelName = loadedSquatchserveModel;
  unloadModelBtn.disabled = true;
  unloadModelBtn.textContent = 'Unloading...';

  try {
    const squatchserveHost = localStorage.getItem('squatchserveHost') || undefined;

    const response = await fetch('/api/squatchserve/unload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: modelName, squatchserveHost })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to unload model');
    }

    addMessage('system', `Model ${modelName} unloaded successfully.`);

    // Immediately refresh status
    await fetchSquatchserveStatus();
  } catch (error) {
    console.error('Error unloading model:', error);
    addMessage('error', `Failed to unload model: ${error.message}`);
  } finally {
    unloadModelBtn.disabled = false;
    unloadModelBtn.textContent = 'Unload';
  }
}

// Start polling for SquatchServe status
function startSquatchserveStatusPolling() {
  // Clear any existing interval
  stopSquatchserveStatusPolling();

  // Fetch immediately
  fetchSquatchserveStatus();

  // Then poll every 30 seconds
  squatchserveStatusInterval = setInterval(fetchSquatchserveStatus, 30000);
}

// Stop polling for SquatchServe status
function stopSquatchserveStatusPolling() {
  if (squatchserveStatusInterval) {
    clearInterval(squatchserveStatusInterval);
    squatchserveStatusInterval = null;
  }
}

// Show/hide model status bar based on provider
function updateModelStatusBarVisibility() {
  if (!modelStatusBar) return;

  if (currentProvider === 'squatchserve') {
    modelStatusBar.style.display = 'flex';
    startSquatchserveStatusPolling();
  } else {
    modelStatusBar.style.display = 'none';
    stopSquatchserveStatusPolling();
    loadedSquatchserveModel = null;
  }
}

// ============ Memory Panel Functions ============

const memoryBtn = document.getElementById('memoryBtn');
const memoryPanel = document.getElementById('memoryPanel');
const memoryPanelClose = document.getElementById('memoryPanelClose');
const memoryPanelOverlay = document.getElementById('memoryPanelOverlay');
const memoryAddFactInput = document.getElementById('memoryAddFactInput');
const memoryAddFactBtn = document.getElementById('memoryAddFactBtn');
const memorySearchInput = document.getElementById('memorySearchInput');
const memorySearchBtn = document.getElementById('memorySearchBtn');

// Cache for loaded cluster member IDs (for edit/delete)
let memoryFactsCache = [];

// Set up memory panel listeners
if (memoryBtn) {
  memoryBtn.addEventListener('click', openMemoryPanel);
}
if (memoryPanelClose) {
  memoryPanelClose.addEventListener('click', closeMemoryPanel);
}
if (memoryPanelOverlay) {
  memoryPanelOverlay.addEventListener('click', closeMemoryPanel);
}
if (memoryAddFactBtn) {
  memoryAddFactBtn.addEventListener('click', addManualFact);
}
if (memoryAddFactInput) {
  memoryAddFactInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') addManualFact();
  });
}
if (memorySearchBtn) {
  memorySearchBtn.addEventListener('click', searchMemory);
}
if (memorySearchInput) {
  memorySearchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') searchMemory();
  });
}

// Tab switching
document.querySelectorAll('.memory-tab').forEach(tab => {
  tab.addEventListener('click', () => switchMemoryTab(tab.dataset.tab));
});

// ---- Initiative bell + panel (things SNH wants to raise) ----
const initiativeBtn = document.getElementById('initiativeBtn');
const initiativeBadge = document.getElementById('initiativeBadge');
const initiativePanel = document.getElementById('initiativePanel');
const initiativePanelOverlay = document.getElementById('initiativePanelOverlay');
const initiativePanelClose = document.getElementById('initiativePanelClose');

if (initiativeBtn) initiativeBtn.addEventListener('click', openInitiativePanel);
if (initiativePanelClose) initiativePanelClose.addEventListener('click', closeInitiativePanel);
if (initiativePanelOverlay) initiativePanelOverlay.addEventListener('click', closeInitiativePanel);

async function refreshInitiativeBadge() {
  if (!initiativeBadge) return;
  try {
    const res = await fetch('/api/memory/initiatives');
    const data = await res.json();
    const total = (data.initiatives || []).length;
    const above = data.aboveThreshold || 0;
    // Badge shows a count whenever ANYTHING is pending; the bell only pulses when
    // something crosses the greeting threshold (worth actively surfacing).
    if (total > 0) {
      initiativeBadge.textContent = total;
      initiativeBadge.style.display = 'inline-flex';
    } else {
      initiativeBadge.style.display = 'none';
    }
    initiativeBtn?.classList.toggle('has-initiatives', above > 0);
  } catch (e) { /* silent */ }
}

function openInitiativePanel() {
  initiativePanel?.classList.add('open');
  initiativePanelOverlay?.classList.add('active');
  loadInitiativeList();
}

function closeInitiativePanel() {
  initiativePanel?.classList.remove('open');
  initiativePanelOverlay?.classList.remove('active');
}

// Compact local timestamp for initiatives, e.g. "7/5 6:42 AM".
function formatInitiativeTime(iso) {
  if (!iso) return '';
  const d = new Date(iso.includes('T') ? iso : iso.replace(' ', 'T') + 'Z');
  if (isNaN(d.getTime())) return '';
  let h = d.getHours();
  const ampm = h >= 12 ? 'PM' : 'AM';
  h = h % 12 || 12;
  const mm = String(d.getMinutes()).padStart(2, '0');
  return `${d.getMonth() + 1}/${d.getDate()} ${h}:${mm} ${ampm}`;
}

async function loadInitiativeList() {
  const container = document.getElementById('initiativeList');
  if (!container) return;
  container.innerHTML = '<div class="memory-loading">Loading…</div>';
  try {
    const res = await fetch('/api/memory/initiatives');
    const data = await res.json();
    const items = data.initiatives || [];
    if (items.length === 0) {
      container.innerHTML = '<div class="memory-empty">Nothing on SNH\'s mind right now.</div>';
      return;
    }
    container.innerHTML = items.map(it => `
      <div class="initiative-item" data-id="${it.id}">
        <div class="initiative-item-head">
          <span class="initiative-type initiative-type-${escapeHtml(it.type)}">${escapeHtml(it.type)}</span>
          <span class="initiative-time" title="${escapeHtml(it.created_at || '')}">${escapeHtml(formatInitiativeTime(it.created_at))}</span>
          <span class="initiative-priority" title="priority">${it.priority}/10</span>
        </div>
        <div class="initiative-content">${escapeHtml(it.content)}</div>
        <div class="initiative-actions">
          <button class="initiative-discuss" data-id="${it.id}">Discuss</button>
          <button class="initiative-dismiss" data-id="${it.id}">Dismiss</button>
        </div>
      </div>
    `).join('');
    container.querySelectorAll('.initiative-dismiss').forEach(btn => {
      btn.addEventListener('click', async () => {
        btn.disabled = true;
        try {
          await fetch(`/api/memory/initiatives/${btn.dataset.id}/dismiss`, { method: 'POST' });
          loadInitiativeList();
          refreshInitiativeBadge();
        } catch (e) { btn.disabled = false; }
      });
    });
    container.querySelectorAll('.initiative-discuss').forEach(btn => {
      btn.addEventListener('click', async () => {
        const actions = btn.closest('.initiative-actions');
        actions?.querySelectorAll('button').forEach(b => (b.disabled = true));
        btn.textContent = 'Opening…';
        try {
          const res = await fetch(`/api/memory/initiatives/${btn.dataset.id}/discuss`, { method: 'POST' });
          const data = await res.json();
          if (!res.ok || !data.conversationId) throw new Error(data.error || 'failed');
          // The initiative is now delivered; refresh sidebar + bell and open the
          // new SNH-initiated conversation so the user can reply.
          await loadConversations();
          await loadConversationById(data.conversationId);
          refreshInitiativeBadge();
          closeInitiativePanel();
        } catch (e) {
          console.error('[Initiatives] Discuss failed:', e);
          btn.textContent = 'Discuss';
          actions?.querySelectorAll('button').forEach(b => (b.disabled = false));
        }
      });
    });
  } catch (error) {
    console.error('[Initiatives] Error loading list:', error);
    container.innerHTML = '<div class="memory-empty">Failed to load</div>';
  }
}

// Keep the bell current: on load and periodically.
refreshInitiativeBadge();
setInterval(refreshInitiativeBadge, 60000);

function openMemoryPanel() {
  memoryPanel?.classList.add('open');
  memoryPanelOverlay?.classList.add('active');
  // Load the active tab data
  const activeTab = document.querySelector('.memory-tab.active');
  if (activeTab) {
    switchMemoryTab(activeTab.dataset.tab);
  } else {
    switchMemoryTab('facts');
  }
}

function closeMemoryPanel() {
  memoryPanel?.classList.remove('open');
  memoryPanelOverlay?.classList.remove('active');
}

function switchMemoryTab(name) {
  // Update tab buttons
  document.querySelectorAll('.memory-tab').forEach(t => t.classList.remove('active'));
  document.querySelector(`.memory-tab[data-tab="${name}"]`)?.classList.add('active');

  // Update tab content
  document.querySelectorAll('.memory-tab-content').forEach(c => c.classList.remove('active'));
  const tabContent = document.getElementById(`memoryTab${name.charAt(0).toUpperCase() + name.slice(1)}`);
  tabContent?.classList.add('active');

  // The Map needs far more room than the other tabs — widen the whole panel.
  memoryPanel?.classList.toggle('map-mode', name === 'map');

  // Load data for the tab
  if (name === 'facts') loadFactsTab();
  else if (name === 'clusters') loadClustersTab();
  else if (name === 'map') loadMapTab();
  else if (name === 'daily') loadDailyTab();
  else if (name === 'self') loadSelfTab();
}

// ---- Facts Tab ----
async function loadFactsTab() {
  const container = document.getElementById('memoryFactsList');
  if (!container) return;
  container.innerHTML = '<div class="memory-loading">Loading facts...</div>';

  try {
    // Load cluster members (facts with IDs for edit/delete)
    const clustersRes = await fetch('/api/memory/clusters');
    const clustersData = await clustersRes.json();
    const clusters = clustersData.clusters || [];

    // Load all members from all clusters
    memoryFactsCache = [];
    for (const cluster of clusters) {
      const clusterRes = await fetch(`/api/memory/clusters/${cluster.id}`);
      const clusterData = await clusterRes.json();
      if (clusterData.members) {
        for (const member of clusterData.members) {
          // Facts tab is user-facts only. Skip any self-observation that may sit
          // inside a user cluster (subject defaults to 'user' for legacy rows).
          if ((member.subject || 'user') !== 'user') continue;
          memoryFactsCache.push({
            id: member.id,
            content: member.content,
            clusterName: cluster.name,
            clusterId: cluster.id
          });
        }
      }
    }

    if (memoryFactsCache.length === 0) {
      container.innerHTML = '<div class="memory-empty">No facts stored yet. Add one above!</div>';
      return;
    }

    container.innerHTML = memoryFactsCache.map(fact => `
      <div class="memory-fact-item" data-id="${fact.id}">
        <div class="memory-fact-content">${escapeHtml(fact.content)}</div>
        <div class="memory-fact-actions">
          <button class="memory-fact-action-btn edit" data-id="${fact.id}" title="Edit">&#9998;</button>
          <button class="memory-fact-action-btn delete" data-id="${fact.id}" title="Delete">&#128465;</button>
        </div>
      </div>
    `).join('');

    // Attach edit/delete handlers
    container.querySelectorAll('.memory-fact-action-btn.edit').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        editFact(btn.dataset.id);
      });
    });
    container.querySelectorAll('.memory-fact-action-btn.delete').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        deleteFact(btn.dataset.id);
      });
    });
  } catch (error) {
    console.error('[MemoryPanel] Error loading facts:', error);
    container.innerHTML = '<div class="memory-empty">Failed to load facts</div>';
  }
}

// ---- Clusters Tab ----
async function loadClustersTab() {
  const container = document.getElementById('memoryClustersList');
  if (!container) return;
  container.innerHTML = '<div class="memory-loading">Loading clusters...</div>';

  try {
    const res = await fetch('/api/memory/clusters');
    const data = await res.json();
    const clusters = data.clusters || [];

    if (clusters.length === 0) {
      container.innerHTML = '<div class="memory-empty">No clusters yet</div>';
      return;
    }

    container.innerHTML = clusters.map(c => `
      <div class="memory-cluster-item" data-id="${c.id}">
        <div class="memory-cluster-header">
          <span class="memory-cluster-name">${escapeHtml(c.name)}</span>
          <span class="memory-cluster-count">${c.member_count}</span>
        </div>
        <div class="memory-cluster-members" id="cluster-members-${c.id}"></div>
      </div>
    `).join('');

    // Attach expand handlers
    container.querySelectorAll('.memory-cluster-header').forEach(header => {
      header.addEventListener('click', () => {
        const item = header.closest('.memory-cluster-item');
        toggleClusterExpand(item.dataset.id);
      });
    });
  } catch (error) {
    console.error('[MemoryPanel] Error loading clusters:', error);
    container.innerHTML = '<div class="memory-empty">Failed to load clusters</div>';
  }
}

async function toggleClusterExpand(clusterId) {
  const membersEl = document.getElementById(`cluster-members-${clusterId}`);
  if (!membersEl) return;

  if (membersEl.classList.contains('expanded')) {
    membersEl.classList.remove('expanded');
    return;
  }

  // Load cluster details
  try {
    const res = await fetch(`/api/memory/clusters/${clusterId}`);
    const data = await res.json();

    let html = '';
    if (data.members) {
      // Clusters tab is user-facts only — never render self-observations that
      // may sit inside a user cluster (legacy rows before the subject guard).
      html += data.members
        .filter(m => (m.subject || 'user') === 'user')
        .map(m => `<div class="memory-cluster-member">${escapeHtml(m.content)}</div>`)
        .join('');
    }

    if (data.linkedClusters && data.linkedClusters.length > 0) {
      html += '<div class="memory-cluster-linked">';
      html += '<div class="memory-cluster-linked-title">Linked clusters:</div>';
      html += data.linkedClusters.map(lc =>
        `<div class="memory-cluster-member">${escapeHtml(lc.name)} (strength: ${lc.strength.toFixed(2)})</div>`
      ).join('');
      html += '</div>';
    }

    membersEl.innerHTML = html;
    membersEl.classList.add('expanded');
  } catch (error) {
    console.error('[MemoryPanel] Error loading cluster details:', error);
  }
}

// ---- Daily Tab ----
async function loadDailyTab() {
  const container = document.getElementById('memoryDailyContent');
  if (!container) return;
  container.innerHTML = '<div class="memory-loading">Loading daily logs...</div>';

  try {
    const res = await fetch('/api/memory');
    const data = await res.json();

    let html = '';

    if (data.dailyToday) {
      html += '<div class="memory-daily-section"><h3>Today</h3>';
      html += `<div class="memory-daily-entry">${escapeHtml(data.dailyToday).replace(/\n/g, '<br>')}</div>`;
      html += '</div>';
    }

    if (data.dailyYesterday) {
      html += '<div class="memory-daily-section"><h3>Yesterday</h3>';
      html += `<div class="memory-daily-entry">${escapeHtml(data.dailyYesterday).replace(/\n/g, '<br>')}</div>`;
      html += '</div>';
    }

    if (!html) {
      html = '<div class="memory-empty">No daily logs found</div>';
    }

    container.innerHTML = html;
  } catch (error) {
    console.error('[MemoryPanel] Error loading daily logs:', error);
    container.innerHTML = '<div class="memory-empty">Failed to load daily logs</div>';
  }
}

// ---- Self Tab (read-only: SNH's self-developed identity) ----
async function loadSelfTab() {
  const container = document.getElementById('memorySelfContent');
  if (!container) return;
  container.innerHTML = '<div class="memory-loading">Loading self...</div>';

  try {
    const res = await fetch('/api/memory/self');
    const data = await res.json();

    const active = data.activeSelfFacts || [];
    const superseded = data.supersededSelfFacts || [];
    const reflections = data.reflections || [];

    const fmtDate = (iso) => {
      if (!iso) return '';
      const d = new Date(iso.includes('T') ? iso : iso.replace(' ', 'T') + 'Z');
      return isNaN(d.getTime()) ? '' : d.toLocaleString();
    };

    let html = '';

    // Injected identity block: seed + active self-facts
    html += '<div class="memory-self-section">';
    html += '<h3>Injected Identity</h3>';
    html += '<div class="memory-self-note">This is exactly what is injected into every chat — a minimal seed plus SNH\'s highest-salience self-observations. Read-only; SNH develops this itself.</div>';
    html += `<div class="memory-self-seed">${escapeHtml(data.seed || '')}</div>`;

    if (active.length === 0) {
      html += '<div class="memory-empty">No self-facts yet. SNH has not observed itself. Reflections build this over time.</div>';
    } else {
      html += '<div class="memory-self-facts">';
      html += active.map(f => `
        <div class="memory-self-fact">
          <div class="memory-self-fact-content">${escapeHtml(f.content)}</div>
          <div class="memory-self-fact-meta">
            <span class="memory-self-salience">salience ${f.salience ?? 5}/10</span>
            <span class="memory-self-when">${escapeHtml(fmtDate(f.created_at))}</span>
          </div>
        </div>
      `).join('');
      html += '</div>';
    }
    html += '</div>';

    // Reflections
    html += '<div class="memory-self-section">';
    html += '<div class="memory-self-section-head"><h3>Recent Reflections</h3><button id="memorySelfReflectBtn" class="memory-self-reflect-btn" title="Run a reflection now">Reflect now</button></div>';
    if (reflections.length === 0) {
      html += '<div class="memory-empty">No reflections yet.</div>';
    } else {
      html += reflections.map(r => {
        const obs = (r.observations || []).map(o => `<li>${escapeHtml(o)}</li>`).join('');
        return `
          <div class="memory-self-reflection">
            <div class="memory-self-reflection-meta">${escapeHtml(fmtDate(r.at))} · ${r.messageCount || 0} msgs · ${r.stored || 0} stored${r.superseded ? `, ${r.superseded} superseded` : ''}</div>
            ${obs ? `<ul class="memory-self-reflection-obs">${obs}</ul>` : '<div class="memory-self-note">Nothing new noticed.</div>'}
          </div>
        `;
      }).join('');
    }
    html += '</div>';

    // Superseded self-facts (identity development history)
    html += '<div class="memory-self-section">';
    html += '<h3>Identity History <span class="memory-self-count">(superseded self-facts)</span></h3>';
    if (superseded.length === 0) {
      html += '<div class="memory-empty">No superseded self-facts. SNH has not revised its self-view yet.</div>';
    } else {
      html += superseded.map(f => `
        <div class="memory-self-fact superseded">
          <div class="memory-self-fact-content">${escapeHtml(f.content)}</div>
          <div class="memory-self-fact-meta">
            <span class="memory-self-salience">salience ${f.salience ?? 5}/10</span>
            <span class="memory-self-when">observed ${escapeHtml(fmtDate(f.created_at))} · revised ${escapeHtml(fmtDate(f.updated_at))}</span>
          </div>
        </div>
      `).join('');
    }
    html += '</div>';

    container.innerHTML = html;

    // "Reflect now" action (triggers the agent; does not edit any self-fact).
    const reflectBtn = document.getElementById('memorySelfReflectBtn');
    if (reflectBtn) {
      reflectBtn.addEventListener('click', async () => {
        reflectBtn.disabled = true;
        reflectBtn.textContent = 'Reflecting…';
        try {
          const r = await fetch('/api/memory/reflect', { method: 'POST' });
          const result = await r.json();
          if (result && result.skipped) {
            reflectBtn.textContent = 'No new conversations';
            setTimeout(() => { reflectBtn.disabled = false; reflectBtn.textContent = 'Reflect now'; }, 2500);
          } else {
            loadSelfTab(); // re-render with the new reflection + self-facts
          }
        } catch (err) {
          console.error('[MemoryPanel] Reflect error:', err);
          reflectBtn.disabled = false;
          reflectBtn.textContent = 'Reflect now';
        }
      });
    }
  } catch (error) {
    console.error('[MemoryPanel] Error loading self:', error);
    container.innerHTML = '<div class="memory-empty">Failed to load self view</div>';
  }
}

// ---- Map Tab (interactive memory constellation) ----
// Renders each cluster as a glowing HUB node (sized by member count, labelled
// with the cluster name); its facts are smaller dots joined to the hub by short
// "spoke" edges, so each cluster reads as a free-floating starburst on dark
// space. Hub-to-hub association links join neighborhoods and light up by link
// strength when a hub is selected. No compound containers — membership is shown
// purely by proximity + spokes. user vs self stay distinct color territories.
// Built on vendored cytoscape.js + fcose.
let mapCy = null;               // cytoscape instance
let mapData = null;             // last /graph payload
let mapSelectedHub = null;      // cluster id of the currently selected hub (or null)
let mapLinkRange = { min: 0.5, max: 1 }; // strength range of rendered hub links
const MAP_AUTO_COLLAPSE_AT = 500; // above this many facts, start collapsed (hubs only)
const FACT_LABEL_MIN_ZOOM_FONT = 10; // fact labels appear only once zoomed in this far
let mapLayoutName = 'cose';     // upgraded to 'fcose' once the extension registers

// Register the fcose layout once. Plain cose/cola let neighborhoods overlap;
// fcose with per-edge ideal lengths keeps members hugging their hub while hubs
// repel each other into distinct constellations.
let fcoseRegistered = false;
function ensureMapLayout() {
  if (fcoseRegistered) return;
  fcoseRegistered = true;
  if (typeof cytoscape !== 'undefined' && typeof cytoscapeFcose !== 'undefined') {
    try {
      cytoscape.use(cytoscapeFcose);
      mapLayoutName = 'fcose';
    } catch (e) {
      // Already registered, or registration failed — keep the cose fallback.
      if (/already/i.test(e.message || '')) mapLayoutName = 'fcose';
      else console.warn('[Map] fcose registration failed, falling back to cose:', e.message);
    }
  } else {
    console.warn('[Map] fcose extension not present — falling back to cose layout.');
  }
}

// Collapse whitespace and truncate to n chars with an ellipsis, for node labels.
function mapTruncate(s, n) {
  s = (s || '').replace(/\s+/g, ' ').trim();
  return s.length > n ? s.slice(0, n - 1).trimEnd() + '…' : s;
}

// Base colors per subject territory (user = blue, self = purple). `fill` is the
// fact-dot color, `hub` a brighter shade for the hub dot, `glow` the halo, and
// `ring` the hub border.
const MAP_COLORS = {
  user: { fill: '#3b82f6', hub: '#60a5fa', glow: 'rgba(59,130,246,0.55)', ring: 'rgba(96,165,250,0.75)' },
  self: { fill: '#a855f7', hub: '#c084fc', glow: 'rgba(168,85,247,0.55)', ring: 'rgba(192,132,252,0.75)' }
};

// Normalize a link strength into 0..1 across the range actually present in the
// rendered graph (falls back to 0.5 when every link has the same weight).
function mapNormStrength(s) {
  const { min, max } = mapLinkRange;
  if (!(max > min)) return 0.5;
  return Math.max(0, Math.min(1, (s - min) / (max - min)));
}

// Map a link strength to a heat color: weak links cool (blue), strong links hot
// (red), a continuous hue sweep between. Scaled from the live weight range so
// the full palette is used no matter how narrow the actual strengths are.
function mapStrengthColor(s) {
  const t = mapNormStrength(s);
  const hue = 210 - 210 * t; // 210° blue (weak) -> 0° red (strong), via cyan/green/amber
  return `hsl(${Math.round(hue)}, 85%, 56%)`;
}

// The Map panel widens via a CSS transition; cytoscape reads the container size
// at init, so resize+fit again now and after the transition settles.
function mapRefit() {
  if (!mapCy) return;
  mapCy.resize();
  mapCy.fit(undefined, 30);
  setTimeout(() => { if (mapCy) { mapCy.resize(); mapCy.fit(undefined, 30); } }, 320);
}

function mapEscape(s) { return escapeHtml(String(s == null ? '' : s)); }

function mapFmtDate(iso) {
  if (!iso) return '—';
  const d = new Date(iso.includes('T') ? iso : iso.replace(' ', 'T') + 'Z');
  return isNaN(d.getTime()) ? '—' : d.toLocaleString();
}

async function loadMapTab() {
  const graphEl = document.getElementById('memoryMapGraph');
  if (!graphEl) return;
  if (typeof cytoscape === 'undefined') {
    graphEl.innerHTML = '<div class="memory-empty">Graph library failed to load.</div>';
    return;
  }
  ensureMapLayout();
  // Already built once — just refit (panel may have been hidden/resized).
  if (mapCy) { mapRefit(); return; }

  graphEl.innerHTML = '<div class="memory-loading">Loading map…</div>';
  try {
    const res = await fetch('/api/memory/graph');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    mapData = await res.json();
    if (!mapData.nodes || mapData.nodes.length === 0) {
      graphEl.innerHTML = '<div class="memory-empty">No memory to map yet.</div>';
      return;
    }
    graphEl.innerHTML = '';

    // Scale handling: above the cap, start collapsed to cluster hubs only (no
    // member facts/spokes) so the constellation stays legible; the Collapse
    // toggle flips the whole graph between hubs-only and full starbursts.
    const collapseBox = document.getElementById('memoryMapCollapse');
    const startCollapsed = mapData.nodes.length > MAP_AUTO_COLLAPSE_AT;
    if (collapseBox) collapseBox.checked = startCollapsed;
    mapSelectedHub = null;

    mapCy = cytoscape({
      container: graphEl,
      style: mapStylesheet(),
      wheelSensitivity: 0.25,
      minZoom: 0.1,
      maxZoom: 3,
      boxSelectionEnabled: false
    });

    wireMapInteractions();
    wireMapControls();
    renderMap();
    mapRefit(); // the panel widens with a CSS transition — resize once it settles
  } catch (err) {
    console.error('[Map] load error:', err);
    graphEl.innerHTML = '<div class="memory-empty">Failed to load map.</div>';
  }
}

function mapStylesheet() {
  return [
    // Hub nodes — the gravitational centre of each cluster. A bright filled dot
    // with a colored halo; the cluster name always rides just below it.
    { selector: 'node.hub', style: {
        'shape': 'ellipse', 'width': 'data(size)', 'height': 'data(size)',
        'background-color': 'data(hub)', 'background-opacity': 1,
        'border-width': 2, 'border-color': 'data(ring)',
        'underlay-color': 'data(glow)', 'underlay-opacity': 0.5, 'underlay-padding': 10,
        'label': 'data(label)', 'font-size': 12, 'font-weight': 700, 'color': '#f1f5f9',
        'text-valign': 'bottom', 'text-halign': 'center', 'text-margin-y': 4,
        'text-wrap': 'wrap', 'text-max-width': 120,
        'text-outline-width': 3, 'text-outline-color': '#05070d', 'text-outline-opacity': 0.95,
        'min-zoomed-font-size': 0, 'z-index': 10
    }},
    // Fact nodes — glowing dots sized by salience. Labels are gated by
    // min-zoomed-font-size so they only appear once the user is zoomed in
    // (or on hover, below), keeping the far-out view clean.
    { selector: 'node.fact', style: {
        'shape': 'ellipse', 'width': 'data(size)', 'height': 'data(size)',
        'background-color': 'data(fill)', 'background-opacity': 1,
        'border-width': 1, 'border-color': 'rgba(255,255,255,0.28)',
        'underlay-color': 'data(glow)', 'underlay-opacity': 0.35, 'underlay-padding': 4,
        'label': 'data(short)', 'font-size': 9, 'color': '#e5e7eb',
        'text-valign': 'bottom', 'text-halign': 'center', 'text-margin-y': 3,
        'text-wrap': 'ellipsis', 'text-max-width': 120,
        'text-outline-width': 2, 'text-outline-color': '#05070d', 'text-outline-opacity': 0.9,
        'text-opacity': 1, 'min-zoomed-font-size': FACT_LABEL_MIN_ZOOM_FONT
    }},
    // Fact label revealed on hover regardless of zoom.
    { selector: 'node.fact.hover', style: {
        'min-zoomed-font-size': 0, 'font-size': 11, 'z-index': 30
    }},
    { selector: 'node.fact:selected', style: {
        'font-size': 11, 'min-zoomed-font-size': 0, 'z-index': 30,
        'border-width': 2, 'border-color': '#fff'
    }},
    // Ghost (superseded) facts — faded, dashed border, dimmer halo.
    { selector: 'node.ghost', style: {
        'background-opacity': 0.25, 'underlay-opacity': 0.12,
        'border-style': 'dashed', 'border-color': 'rgba(255,255,255,0.4)'
    }},
    // Facts with a pending question — amber halo ring (overrides the base glow).
    { selector: 'node.pending', style: {
        'underlay-color': '#fbbf24', 'underlay-opacity': 0.6, 'underlay-padding': 7
    }},
    // Selected hub — white rim + brighter halo, lifted above its neighborhood.
    { selector: 'node.hub.hubsel', style: {
        'border-width': 3, 'border-color': '#ffffff',
        'underlay-opacity': 0.8, 'z-index': 40
    }},
    // Spoke edges (hub -> its own facts): tinted by subject, very low opacity so
    // the dots dominate and the rays only imply membership.
    { selector: 'edge.spoke', style: {
        'width': 1, 'line-color': 'data(fill)', 'opacity': 0.16, 'curve-style': 'straight'
    }},
    // Hub-to-hub association links: neutral slate, a touch brighter than spokes.
    // Recolored by strength on hub selection (see selectHub).
    { selector: 'edge.clink', style: {
        'width': 'data(w)', 'line-color': 'rgba(148,163,184,0.5)',
        'opacity': 0.5, 'curve-style': 'bezier'
    }},
    // Supersede edges (belief history) — directed, dashed.
    { selector: 'edge.supersede', style: {
        'width': 1.5, 'line-color': '#f59e0b', 'line-style': 'dashed',
        'target-arrow-color': '#f59e0b', 'target-arrow-shape': 'triangle',
        'curve-style': 'bezier', 'arrow-scale': 0.8, 'opacity': 0.7
    }},
    // Dimming: `.faded` for hub-selection focus, `.dim` for search.
    { selector: '.faded', style: { 'opacity': 0.07 } },
    { selector: '.dim', style: { 'opacity': 0.1 } },
    { selector: 'node.match', style: {
        'border-width': 3, 'border-color': '#22d3ee', 'opacity': 1
    }},
    { selector: '.hidden', style: { 'display': 'none' } }
  ];
}

// Salience 1..10 -> fact-dot diameter (kept smaller than hubs).
function mapFactSize(sal) {
  const s = Math.max(1, Math.min(10, Number(sal) || 5));
  return 10 + s * 2.2; // 12.2..32
}

// Member count -> hub diameter (sqrt so a 26-fact cluster stays proportionate).
function mapHubSize(count) {
  return 26 + Math.min(40, Math.sqrt(Math.max(1, count)) * 7); // ~33..~62
}

// Build the element set for the current mode and hand it to cytoscape.
function renderMap() {
  if (!mapCy || !mapData) return;
  const collapsed = document.getElementById('memoryMapCollapse')?.checked;
  const showGhosts = document.getElementById('memoryMapGhosts')?.checked !== false;

  const clusterById = new Map(mapData.clusters.map(c => [c.id, c]));
  const els = [];
  const shownFactIds = new Set();

  // One hub dot per cluster — always present, sized by member count.
  for (const c of mapData.clusters) {
    const col = MAP_COLORS[c.subject] || MAP_COLORS.user;
    els.push({ data: {
      id: c.id, clusterId: c.id, kind: 'hub', subject: c.subject,
      label: c.name || 'cluster', size: mapHubSize(c.total),
      hub: col.hub, ring: col.ring, glow: col.glow
    }, classes: 'hub' });
  }

  // Facts + spokes — skipped entirely in collapsed (hubs-only) mode.
  if (!collapsed) {
    for (const n of mapData.nodes) {
      if (!n.clusterId || !clusterById.has(n.clusterId)) continue; // skip orphans
      const isGhost = n.status === 'superseded' || !!n.supersededBy;
      if (isGhost && !showGhosts) continue;
      const col = MAP_COLORS[n.subject] || MAP_COLORS.user;
      let cls = 'fact';
      if (isGhost) cls += ' ghost';
      if (n.pendingQuestions > 0) cls += ' pending';
      els.push({ data: {
        id: n.id, clusterId: n.clusterId, kind: 'fact', subject: n.subject,
        size: mapFactSize(n.salience), fill: col.fill, glow: col.glow,
        short: mapTruncate(n.content, 40)
      }, classes: cls });
      shownFactIds.add(n.id);
      // Spoke: hub -> fact. Tinted by the fact's subject.
      els.push({ data: {
        id: `spoke-${n.clusterId}-${n.id}`, source: n.clusterId, target: n.id,
        kind: 'spoke', fill: col.fill
      }, classes: 'spoke' });
    }
  }

  // Supersede edges (only when both endpoints are on screen).
  for (const e of mapData.edges) {
    if (e.type !== 'supersede') continue;
    if (shownFactIds.has(e.source) && shownFactIds.has(e.target)) {
      els.push({ data: { id: `sup-${e.source}-${e.target}`, source: e.source, target: e.target }, classes: 'supersede' });
    }
  }

  // Hub-to-hub association links. Track the live strength range so link colors
  // and widths scale across the actual weights present.
  let mn = Infinity, mx = -Infinity;
  const clinks = [];
  for (const e of mapData.edges) {
    if (e.type !== 'cluster-link') continue;
    if (clusterById.has(e.source) && clusterById.has(e.target)) {
      const s = e.strength || 0.5;
      mn = Math.min(mn, s); mx = Math.max(mx, s);
      clinks.push(e);
    }
  }
  mapLinkRange = { min: isFinite(mn) ? mn : 0.5, max: isFinite(mx) ? mx : 1 };
  for (const e of clinks) {
    const s = e.strength || 0.5;
    els.push({ data: {
      id: `clink-${e.source}-${e.target}`, source: e.source, target: e.target,
      strength: s, w: 0.6 + 2.4 * mapNormStrength(s)
    }, classes: 'clink' });
  }

  mapSelectedHub = null;
  mapCy.elements().remove();
  mapCy.add(els);
  runMapLayout();
  applyMapSearch();
}

function runMapLayout() {
  const opts = mapLayoutName === 'fcose'
    ? {
        name: 'fcose',
        quality: 'proof',       // best separation quality (offline, one-shot)
        animate: false,
        randomize: true,        // fresh global layout, not stuck on prior positions
        // Hubs repel each other hard so neighborhoods fly apart; facts repel
        // only mildly so a hub's members stay a tight starburst around it.
        nodeRepulsion: node => node.hasClass('hub') ? 120000 : 6000,
        // Short ideal length on spokes pulls members onto their hub; long on
        // hub-to-hub links keeps associated constellations distinct but nearer
        // than unlinked ones.
        idealEdgeLength: edge => edge.hasClass('spoke') ? 46 : 300,
        edgeElasticity: edge => edge.hasClass('spoke') ? 0.55 : 0.08,
        gravity: 0.12,
        gravityRange: 3.8,
        // user vs self have no cross-links, so they land as separate packed
        // components; separateMapZones then splits them left/right.
        tile: true,
        packComponents: true,
        nodeSeparation: 120,
        padding: 50,
        numIter: 2600,
        stop: () => { separateNeighbourhoods(); separateMapZones(); mapCy.fit(undefined, 45); }
      }
    : {
        name: 'cose', animate: false, randomize: true,
        nodeRepulsion: 16000, idealEdgeLength: 55, gravity: 0.35,
        componentSpacing: 140, padding: 40, nodeOverlap: 14,
        coolingFactor: 0.95, numIter: 1400,
        stop: () => { separateNeighbourhoods(); separateMapZones(); mapCy.fit(undefined, 45); }
      };
  mapCy.layout(opts).run();
  // fcose with animate:false still fires `stop` synchronously after run(); fit
  // there. Guard with an immediate fit for the cose path too.
  mapCy.fit(undefined, 45);
}

// Guarantee dark space between constellations. fcose spreads hubs well but two
// starbursts (a hub + its member ring) can still overlap — especially a large
// cluster next to a linked one. This treats each cluster as a group (hub + its
// facts), models it as a circle centred on the hub, and iteratively pushes any
// overlapping pair apart along the line between their centres until every pair
// has at least `pad` px of gap. Deterministic and convergent for ~dozens of
// clusters. Groups move as a whole so the intra-cluster starburst is preserved.
function separateNeighbourhoods(pad = 28, iterations = 80) {
  if (!mapCy) return;
  const groups = [];
  mapCy.nodes('.hub').forEach(h => {
    const cid = h.data('clusterId');
    const facts = mapCy.nodes('.fact').filter(n => n.data('clusterId') === cid);
    const members = h.union(facts);
    const c = h.position();
    let r = h.width() / 2;
    facts.forEach(f => { r = Math.max(r, Math.hypot(f.position('x') - c.x, f.position('y') - c.y) + f.width() / 2); });
    groups.push({ members, c: { x: c.x, y: c.y }, r });
  });
  if (groups.length < 2) return;
  for (let it = 0; it < iterations; it++) {
    let moved = false;
    for (let i = 0; i < groups.length; i++) {
      for (let j = i + 1; j < groups.length; j++) {
        const A = groups[i], B = groups[j];
        let dx = B.c.x - A.c.x, dy = B.c.y - A.c.y;
        let d = Math.hypot(dx, dy);
        const need = A.r + B.r + pad;
        if (d >= need) continue;
        if (d < 0.01) { dx = 1; dy = 0; d = 1; } // coincident — pick an axis
        const push = (need - d) / 2;
        const ux = dx / d, uy = dy / d;
        A.members.positions(n => ({ x: n.position('x') - ux * push, y: n.position('y') - uy * push }));
        B.members.positions(n => ({ x: n.position('x') + ux * push, y: n.position('y') + uy * push }));
        A.c.x -= ux * push; A.c.y -= uy * push;
        B.c.x += ux * push; B.c.y += uy * push;
        moved = true;
      }
    }
    if (!moved) break;
  }
}

// Separate the self (purple) and user (blue) territories into two side-by-side
// zones. fcose packs each subject as its own component but can place them
// anywhere; this shifts every self node left so the self zone sits entirely left
// of the user zone with a clear gap. Uniform translation preserves the intra-
// zone arrangement (no new overlaps) and is deterministic (verifiable from
// positions). Every element is now a leaf, so we move all self nodes directly.
function separateMapZones(gap = 200) {
  if (!mapCy) return;
  const self = mapCy.nodes('[subject = "self"]');
  const user = mapCy.nodes('[subject = "user"]');
  if (self.empty() || user.empty()) return;
  const sbb = self.boundingBox();
  const ubb = user.boundingBox();
  // Target: self.x2 == user.x1 - gap  (self entirely left of user).
  const dx = (ubb.x1 - gap) - sbb.x2;
  if (Math.abs(dx) < 1) return;
  self.positions(n => ({ x: n.position('x') + dx, y: n.position('y') }));
}

function wireMapControls() {
  const ghosts = document.getElementById('memoryMapGhosts');
  const collapse = document.getElementById('memoryMapCollapse');
  const fit = document.getElementById('memoryMapFit');
  const search = document.getElementById('memoryMapSearch');

  ghosts?.addEventListener('change', () => renderMap());
  collapse?.addEventListener('change', () => renderMap());
  fit?.addEventListener('click', () => mapCy && mapCy.fit(undefined, 30));
  search?.addEventListener('input', applyMapSearch);
}

function wireMapInteractions() {
  // Click a fact -> detail (and drop any hub focus). Click a hub -> focus its
  // neighborhood + light its links. Click empty space -> clear.
  mapCy.on('tap', 'node.fact', (evt) => { clearHubSelection(); showFactDetail(evt.target.id()); });
  mapCy.on('tap', 'node.hub', (evt) => selectHub(evt.target.data('clusterId')));
  mapCy.on('tap', (evt) => { if (evt.target === mapCy) { clearHubSelection(); hideMapDetail(); } });
  // Hover reveals a fact's label regardless of zoom.
  mapCy.on('mouseover', 'node.fact', (evt) => evt.target.addClass('hover'));
  mapCy.on('mouseout', 'node.fact', (evt) => evt.target.removeClass('hover'));
}

// The set of graph elements belonging to one cluster: its hub, its facts, and
// the spokes joining them.
function mapNeighbourhood(cid) {
  const hub = mapCy.getElementById(cid);
  const facts = mapCy.nodes('.fact').filter(n => n.data('clusterId') === cid);
  const spokes = mapCy.edges('.spoke').filter(e => e.data('source') === cid);
  return hub.union(facts).union(spokes);
}

// Strip any inline strength styling previously painted onto hub-hub links.
function resetClinkStyles() {
  if (!mapCy) return;
  mapCy.edges('.clink').removeStyle('line-color opacity width z-index');
}

// Drop hub focus: un-dim everything and restore default link styling.
function clearHubSelection() {
  if (!mapCy) return;
  mapSelectedHub = null;
  resetClinkStyles();
  mapCy.elements().removeClass('faded hubsel');
}

// Focus a cluster hub: dim everything, then light its own neighborhood plus each
// linked cluster, coloring the connecting hub-hub links by strength (hot = strong,
// cool = weak). Opens the summary sidebar with the linked clusters ranked.
function selectHub(clusterId) {
  if (!mapCy) return;
  const hub = mapCy.getElementById(clusterId);
  if (hub.empty()) return;

  // Selection and search share the visual channel — clear search first.
  const searchEl = document.getElementById('memoryMapSearch');
  if (searchEl && searchEl.value) searchEl.value = '';
  mapCy.elements().removeClass('dim match');
  resetClinkStyles();

  mapSelectedHub = clusterId;
  mapCy.batch(() => {
    mapCy.elements().addClass('faded');
    let lit = mapNeighbourhood(clusterId);
    hub.addClass('hubsel');

    const linked = [];
    mapCy.edges('.clink').forEach(e => {
      const s = e.data('source'), t = e.data('target');
      if (s !== clusterId && t !== clusterId) return;
      const other = s === clusterId ? t : s;
      const strength = e.data('strength');
      const col = mapStrengthColor(strength);
      e.style({ 'line-color': col, 'opacity': 0.95,
        'width': 1.5 + 3 * mapNormStrength(strength), 'z-index': 25 });
      lit = lit.union(e).union(mapNeighbourhood(other));
      linked.push({ id: other, strength });
    });

    lit.removeClass('faded');
    linked.sort((a, b) => b.strength - a.strength);
    showClusterDetail(clusterId, linked);
  });
}

function applyMapSearch() {
  if (!mapCy) return;
  const q = (document.getElementById('memoryMapSearch')?.value || '').trim().toLowerCase();
  mapCy.batch(() => {
    if (q && mapSelectedHub) clearHubSelection(); // search overrides hub focus
    mapCy.elements().removeClass('dim match');
    if (!q) return;
    const byId = new Map(mapData.nodes.map(n => [n.id, n]));
    const matches = mapCy.nodes('.fact').filter(n => {
      const rec = byId.get(n.id());
      return rec && rec.content && rec.content.toLowerCase().includes(q);
    });
    if (matches.length === 0) return;
    mapCy.elements().addClass('dim');
    matches.removeClass('dim').addClass('match');
    // Keep each matched fact's hub + spoke visible for context.
    matches.forEach(n => mapCy.getElementById(n.data('clusterId')).removeClass('dim'));
    matches.connectedEdges('.spoke').removeClass('dim');
  });
}

function showFactDetail(id) {
  const n = mapData?.nodes.find(x => x.id === id);
  if (!n) return;
  const cluster = mapData.clusters.find(c => c.id === n.clusterId);
  const isGhost = n.status === 'superseded' || !!n.supersededBy;
  const replacement = n.supersededBy && mapData.nodes.find(x => x.id === n.supersededBy);
  const replaces = mapData.nodes.filter(x => x.supersededBy === n.id);

  let html = `<button class="memory-map-detail-close">&times;</button>`;
  html += `<div class="mmd-subject mmd-${mapEscape(n.subject)}">${mapEscape(n.subject)} fact</div>`;
  html += `<div class="mmd-content">${mapEscape(n.content)}</div>`;
  html += `<dl class="mmd-fields">`;
  html += `<dt>Salience</dt><dd>${mapEscape(n.salience)}/10</dd>`;
  html += `<dt>Status</dt><dd>${isGhost ? 'superseded (ghost)' : mapEscape(n.status)}</dd>`;
  html += `<dt>Cluster</dt><dd>${mapEscape(cluster ? cluster.name : '—')}</dd>`;
  html += `<dt>Source</dt><dd>${mapEscape(n.source || '—')}</dd>`;
  html += `<dt>Created</dt><dd>${mapEscape(mapFmtDate(n.createdAt))}</dd>`;
  html += `<dt>Updated</dt><dd>${mapEscape(mapFmtDate(n.updatedAt))}</dd>`;
  if (n.pendingQuestions > 0) html += `<dt>Pending Q</dt><dd>${n.pendingQuestions} open</dd>`;
  html += `</dl>`;
  if (replacement) html += `<div class="mmd-rel">↳ Replaced by: “${mapEscape(replacement.content)}”</div>`;
  if (replaces.length) html += `<div class="mmd-rel">↑ Replaces: ${replaces.map(r => `“${mapEscape(r.content)}”`).join(', ')}</div>`;
  openMapDetail(html);
}

// All clusters linked to `id`, sorted strongest-first.
function mapLinkedClusters(id) {
  const out = [];
  for (const e of (mapData?.edges || [])) {
    if (e.type !== 'cluster-link') continue;
    if (e.source === id) out.push({ id: e.target, strength: e.strength || 0.5 });
    else if (e.target === id) out.push({ id: e.source, strength: e.strength || 0.5 });
  }
  out.sort((a, b) => b.strength - a.strength);
  return out;
}

function showClusterDetail(id, linked) {
  const c = mapData?.clusters.find(x => x.id === id);
  if (!c) return;
  if (!linked) linked = mapLinkedClusters(id);
  let html = `<button class="memory-map-detail-close">&times;</button>`;
  html += `<div class="mmd-subject mmd-${mapEscape(c.subject)}">${mapEscape(c.subject)} cluster</div>`;
  html += `<div class="mmd-content">${mapEscape(c.name)}</div>`;
  if (c.description) html += `<div class="mmd-desc">${mapEscape(c.description)}</div>`;
  html += `<dl class="mmd-fields">`;
  html += `<dt>Facts</dt><dd>${c.total}</dd>`;
  html += `<dt>Active</dt><dd>${c.active}</dd>`;
  html += `<dt>Superseded</dt><dd>${c.superseded}</dd>`;
  html += `</dl>`;
  // Linked clusters, ranked by strength, color-coded to match the lit links.
  if (linked.length) {
    html += `<div class="mmd-links"><div class="mmd-links-title">Linked clusters</div>`;
    for (const l of linked) {
      const lc = mapData.clusters.find(x => x.id === l.id);
      const col = mapStrengthColor(l.strength);
      html += `<div class="mmd-link-row">`
        + `<span class="mmd-link-dot" style="background:${col};color:${col}"></span>`
        + `<span class="mmd-link-name">${mapEscape(lc ? lc.name : '—')}</span>`
        + `<span class="mmd-link-strength" style="color:${col}">${l.strength.toFixed(2)}</span>`
        + `</div>`;
    }
    html += `</div>`;
  }
  openMapDetail(html);
}

function openMapDetail(html) {
  const box = document.getElementById('memoryMapDetail');
  if (!box) return;
  box.innerHTML = html;
  box.classList.add('open');
  box.querySelector('.memory-map-detail-close')?.addEventListener('click', hideMapDetail);
}

function hideMapDetail() {
  const box = document.getElementById('memoryMapDetail');
  if (box) { box.classList.remove('open'); box.innerHTML = ''; }
  mapCy?.$(':selected').unselect();
  clearHubSelection();
}

// ---- Search Tab ----
async function searchMemory() {
  const query = memorySearchInput?.value.trim();
  if (!query) return;

  const resultsContainer = document.getElementById('memorySearchResults');
  if (!resultsContainer) return;
  resultsContainer.innerHTML = '<div class="memory-loading">Searching...</div>';

  try {
    const res = await fetch('/api/memory/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, limit: 15 })
    });

    const data = await res.json();
    const results = data.results || [];

    if (results.length === 0) {
      resultsContainer.innerHTML = '<div class="memory-empty">No results found</div>';
      return;
    }

    resultsContainer.innerHTML = results.map(r => `
      <div class="memory-search-result">
        <div>${escapeHtml(r.text?.substring(0, 300) || '')}${(r.text?.length || 0) > 300 ? '...' : ''}</div>
        <div class="memory-search-result-score">
          Score: ${(r.similarity || 0).toFixed(3)}
          <span class="memory-search-result-source ${r.source || ''}">${r.source || 'unknown'}</span>
        </div>
      </div>
    `).join('');
  } catch (error) {
    console.error('[MemoryPanel] Error searching:', error);
    resultsContainer.innerHTML = '<div class="memory-empty">Search failed</div>';
  }
}

// ---- Add Fact ----
async function addManualFact() {
  const input = memoryAddFactInput;
  if (!input) return;
  const fact = input.value.trim();
  if (!fact) return;

  input.value = '';
  input.disabled = true;

  try {
    const res = await fetch('/api/memory/add', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fact })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Failed to add fact');
    }

    const data = await res.json();
    console.log('[MemoryPanel] Fact added:', data);

    // Refresh facts tab
    loadFactsTab();
  } catch (error) {
    console.error('[MemoryPanel] Error adding fact:', error);
    alert('Failed to add fact: ' + error.message);
  } finally {
    input.disabled = false;
    input.focus();
  }
}

// ---- Edit Fact ----
async function editFact(memberId) {
  const fact = memoryFactsCache.find(f => f.id === memberId);
  if (!fact) return;

  const newContent = prompt('Edit fact:', fact.content);
  if (!newContent || newContent.trim() === fact.content) return;

  try {
    const res = await fetch('/api/memory/edit', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ memberId, content: newContent.trim() })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Failed to edit fact');
    }

    loadFactsTab();
  } catch (error) {
    console.error('[MemoryPanel] Error editing fact:', error);
    alert('Failed to edit fact: ' + error.message);
  }
}

// ---- Delete Fact ----
async function deleteFact(memberId) {
  const fact = memoryFactsCache.find(f => f.id === memberId);
  if (!fact) return;

  if (!confirm(`Delete this fact?\n\n"${fact.content.substring(0, 100)}..."`)) return;

  try {
    const res = await fetch(`/api/memory/fact/${memberId}`, {
      method: 'DELETE'
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || 'Failed to delete fact');
    }

    loadFactsTab();
  } catch (error) {
    console.error('[MemoryPanel] Error deleting fact:', error);
    alert('Failed to delete fact: ' + error.message);
  }
}

// ---- Config Section Builder (shared by unified settings tabs) ----

function createConfigSection(title, fields) {
  const section = document.createElement('div');
  section.className = 'config-section';
  if (title) {
    const h3 = document.createElement('h3');
    h3.textContent = title;
    section.appendChild(h3);
  }

  for (const field of fields) {
    const item = document.createElement('div');
    item.className = 'config-item';

    // Fix 7: Generate unique ID and connect label to input
    const inputId = `settings-field-${field.key.replace(/\./g, '-')}`;

    const label = document.createElement('label');
    label.textContent = field.label;
    label.setAttribute('for', inputId);
    item.appendChild(label);

    let input;
    if (field.type === 'checkbox') {
      input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = !!field.value;
      input.dataset.configKey = field.key;
    } else if (field.type === 'select') {
      input = document.createElement('select');
      for (const opt of field.options) {
        const option = document.createElement('option');
        option.value = opt;
        option.textContent = opt;
        if (opt === field.value) option.selected = true;
        input.appendChild(option);
      }
      input.dataset.configKey = field.key;
    } else {
      input = document.createElement('input');
      input.type = field.type || 'text';
      input.value = field.value ?? '';
      if (field.type === 'number' && field.step) input.step = field.step;
      input.dataset.configKey = field.key;
    }

    // Fix 7: Set the matching ID on the input element
    input.id = inputId;

    item.appendChild(input);
    section.appendChild(item);
  }

  return section;
}

