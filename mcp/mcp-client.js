/**
 * MCP (Model Context Protocol) Client Manager
 * Maintains a registry of tool servers and provides methods to:
 * - Get all available tools in OpenAI function calling format
 * - Execute tool calls by routing to the correct tool implementation
 */

const fs = require('fs');
const path = require('path');
const SearXNGTool = require('./tools/searxng');
const WebFetchTool = require('./tools/web-fetch');

class MCPClient {
  constructor() {
    this.tools = new Map(); // tool name -> tool instance
    this.config = null;
  }

  /**
   * Load tool configuration from tools.json and initialize enabled tools
   */
  loadConfig(configPath) {
    const resolvedPath = configPath || path.join(__dirname, 'tools.json');
    console.log(`MCP: Loading config from ${resolvedPath}`);
    try {
      const raw = fs.readFileSync(resolvedPath, 'utf-8');
      this.config = JSON.parse(raw);
      console.log(`MCP: Config loaded, ${this.config.tools?.length || 0} tool(s) defined`);
    } catch (error) {
      console.warn('MCP: Could not load tools.json:', error.message);
      this.config = { tools: [] };
    }

    this.tools.clear();

    for (const toolConfig of this.config.tools) {
      if (!toolConfig.enabled) {
        console.log(`MCP: Skipping disabled tool "${toolConfig.name}"`);
        continue;
      }

      if (toolConfig.name === 'searxng') {
        const tool = new SearXNGTool(toolConfig.endpoint);
        this.tools.set(tool.name, tool);
        console.log(`MCP: Registered tool "${tool.name}" -> endpoint ${toolConfig.endpoint}`);
      }
      // web_fetch is always available when tools are enabled
    }

    // web_fetch is always available when any tool is enabled
    if (this.tools.size > 0 && !this.tools.has('web_fetch')) {
      const webFetch = new WebFetchTool();
      this.tools.set(webFetch.name, webFetch);
      console.log('MCP: Registered built-in tool "web_fetch"');
    }

    console.log(`MCP: ${this.tools.size} tool(s) ready: [${this.getToolNames().join(', ')}]`);
  }

  /**
   * Get all available tools formatted for OpenAI function calling
   * Returns array suitable for the "tools" parameter in chat completions
   */
  getToolsForOpenAI() {
    const specs = [];
    for (const tool of this.tools.values()) {
      specs.push(tool.getOpenAIFunctionSpec());
    }
    return specs;
  }

  /**
   * Execute a tool call by name
   * @param {string} toolName - The tool function name
   * @param {Object} args - The parsed arguments for the tool
   * @param {Object} context - Optional context (e.g., { searxngHost } for endpoint override)
   * @returns {Object} Tool execution result
   */
  async executeTool(toolName, args, context = {}) {
    const tool = this.tools.get(toolName);
    if (!tool) {
      return { error: `Unknown tool: ${toolName}` };
    }

    try {
      // Pass endpoint override for tools that support it
      if (toolName === 'web_search' && context.searxngHost) {
        return await tool.execute(args, context.searxngHost);
      }
      return await tool.execute(args);
    } catch (error) {
      return { error: `Tool execution failed: ${error.message}` };
    }
  }

  /**
   * Check if any tools are registered
   */
  hasTools() {
    return this.tools.size > 0;
  }

  /**
   * Get list of registered tool names
   */
  getToolNames() {
    return Array.from(this.tools.keys());
  }
}

module.exports = MCPClient;
