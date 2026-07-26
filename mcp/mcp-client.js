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
const CreateCronJobTool = require('./tools/create-cron-job');
const { getConfig } = require('../db/config');

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

    // web_fetch rides along with SEARCH tools specifically — fetching a page is
    // only useful when there is a search that produced the URL. Note this checks
    // for a search tool rather than "any tool at all": action tools register
    // below and must not drag web_fetch on with them.
    if (this.tools.has('web_search') && !this.tools.has('web_fetch')) {
      const webFetch = new WebFetchTool();
      this.tools.set(webFetch.name, webFetch);
      console.log('MCP: Registered built-in tool "web_fetch"');
    }

    // Action tools register independently of the search stack. create_cron_job
    // is gated only on its own config flag — it must be available when SearXNG
    // is off, which is the default.
    const cronCfg = (getConfig().tools && getConfig().tools.cron) || {};
    if (cronCfg.enabled !== false) {
      const cronTool = new CreateCronJobTool();
      this.tools.set(cronTool.name, cronTool);
      console.log(`MCP: Registered action tool "${cronTool.name}" (tier=${cronTool.tier}, propose-only)`);
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
      // Everything else gets the context object as its second argument. Action
      // tools need it (create_cron_job records which conversation proposed the
      // job); web_fetch takes only args and ignores it.
      return await tool.execute(args, context);
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

  /** Is one specific tool registered? */
  hasTool(name) {
    return this.tools.has(name);
  }

  /**
   * Tier metadata for every registered tool that declares it. Nothing reads this
   * yet — the tool registry will. Tools without getTierMetadata() are omitted
   * rather than guessed at.
   */
  getTierMetadata() {
    const out = [];
    for (const tool of this.tools.values()) {
      if (typeof tool.getTierMetadata === 'function') out.push(tool.getTierMetadata());
    }
    return out;
  }

  /**
   * Get list of registered tool names
   */
  getToolNames() {
    return Array.from(this.tools.keys());
  }
}

module.exports = MCPClient;
