// Bridge relay rejections into Qwen's native tool-result/history path.
// No scheduler, retry, tool permission, or successful invocation is changed.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const chunks = path.join(path.dirname(path.resolve(process.env.AGENT_COMPANY_QWEN_CLI)), 'chunks');
let baseTool;
for (const entry of fs.readdirSync(chunks)) {
  if (!entry.endsWith('.js')) continue;
  const file = path.join(chunks, entry);
  if (!fs.readFileSync(file, 'utf8').includes('var BaseDeclarativeTool = class')) continue;
  const loaded = await import(pathToFileURL(file).href);
  if (loaded.BaseDeclarativeTool) { baseTool = loaded.BaseDeclarativeTool; break; }
}
if (!baseTool || typeof baseTool.prototype.build !== 'function') {
  throw new Error('Installed Qwen does not expose the required native tool-validation path');
}
const build = baseTool.prototype.build;
baseTool.prototype.build = function (params) {
  if (params && typeof params === 'object' && Object.hasOwn(params, '__agent_company_tool_error')) {
    // Throw before native parameter normalization, permissions or side effects.
    // Both native master and worker loops retain this as a failed tool result.
    throw new Error(String(params.__agent_company_tool_error));
  }
  return build.call(this, params);
};
