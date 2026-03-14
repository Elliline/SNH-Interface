/**
 * Web Fetch Tool
 * Fetches a URL and returns plain text content (HTML tags stripped)
 */

class WebFetchTool {
  constructor() {
    this.name = 'web_fetch';
    this.description = 'Fetch a web page by URL and return its text content. Use this to read the full content of a page found via web_search.';
    this.parameters = {
      type: 'object',
      properties: {
        url: {
          type: 'string',
          description: 'The URL to fetch'
        }
      },
      required: ['url']
    };
  }

  getOpenAIFunctionSpec() {
    return {
      type: 'function',
      function: {
        name: this.name,
        description: this.description,
        parameters: this.parameters
      }
    };
  }

  /**
   * Validate URL: must be http/https and not a private IP
   */
  _isPublicUrl(urlStr) {
    try {
      const url = new URL(urlStr);
      if (url.protocol !== 'http:' && url.protocol !== 'https:') return false;

      const hostname = url.hostname;
      // Block localhost and loopback
      if (hostname === 'localhost' || hostname === '127.0.0.1' || hostname === '::1') return false;
      // Block private RFC 1918 ranges
      if (/^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(hostname)) return false;
      if (/^172\.(1[6-9]|2[0-9]|3[0-1])\.\d{1,3}\.\d{1,3}$/.test(hostname)) return false;
      if (/^192\.168\.\d{1,3}\.\d{1,3}$/.test(hostname)) return false;
      // Block link-local
      if (/^169\.254\.\d{1,3}\.\d{1,3}$/.test(hostname)) return false;

      return true;
    } catch (e) {
      return false;
    }
  }

  /**
   * Strip HTML tags and return plain text
   */
  _htmlToText(html) {
    return html
      .replace(/<script[\s\S]*?<\/script>/gi, '')
      .replace(/<style[\s\S]*?<\/style>/gi, '')
      .replace(/<[^>]+>/g, ' ')
      .replace(/&nbsp;/g, ' ')
      .replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&quot;/g, '"')
      .replace(/&#39;/g, "'")
      .replace(/\s+/g, ' ')
      .trim();
  }

  async execute(args) {
    const { url } = args;

    if (!url || typeof url !== 'string') {
      return { error: 'Missing or invalid URL' };
    }

    if (!this._isPublicUrl(url)) {
      return { error: 'URL must be a public http/https address' };
    }

    try {
      const response = await fetch(url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (compatible; SquatchNeuroHub/1.0)',
          'Accept': 'text/html,application/xhtml+xml,text/plain'
        },
        redirect: 'follow',
        signal: AbortSignal.timeout(10000)
      });

      if (!response.ok) {
        return { error: `Fetch failed with status ${response.status}` };
      }

      const html = await response.text();
      const text = this._htmlToText(html);
      const truncated = text.substring(0, 4000);

      return {
        url,
        content: truncated,
        truncated: text.length > 4000
      };
    } catch (error) {
      return { error: `Fetch failed: ${error.message}` };
    }
  }
}

module.exports = WebFetchTool;
