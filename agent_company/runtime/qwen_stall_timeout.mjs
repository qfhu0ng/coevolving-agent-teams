// Preserve the previously verified long-response fix without changing native
// delivery/continuation. Discover the exported TeamManager in the installed CLI.
import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
const seconds = Number(process.env.QWEN_TEAM_STALL_TIMEOUT_SECONDS);
if (!Number.isFinite(seconds) || seconds <= 0) throw new Error('Invalid Qwen Team inactivity timeout');
const chunks = path.join(path.dirname(path.resolve(process.env.AGENT_COMPANY_QWEN_CLI)), 'chunks');
let manager;
for (const entry of fs.readdirSync(chunks)) {
  if (!entry.endsWith('.js')) continue;
  const file = path.join(chunks, entry);
  const source = fs.readFileSync(file, 'utf8');
  if (!source.includes('static STALL_THRESHOLD_S')) continue;
  const loaded = await import(pathToFileURL(file).href);
  if (loaded.TeamManager) { manager = loaded.TeamManager; break; }
}
if (!manager || typeof manager.STALL_THRESHOLD_S !== 'number') {
  throw new Error('Installed Qwen does not expose the verified Team inactivity setting');
}
manager.STALL_THRESHOLD_S = seconds;
process.stderr.write(`Agent Company: native Team inactivity timeout ${seconds}s\n`);
