'use strict';
// Isolate Qwen startup settings and .env discovery. This is deliberately not
// a filesystem sandbox and does not change the native Team runtime.
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const { fileURLToPath } = require('node:url');
const { syncBuiltinESMExports } = require('node:module');
// Observe the actual process, not the host architecture or distribution name.
process.stderr.write('Agent Company: Node runtime ' + JSON.stringify({
  version: process.version, arch: process.arch, platform: process.platform,
  libuv: process.versions.uv, execPath: process.execPath,
}) + '\n');
const runHome = process.env.AGENT_COMPANY_QWEN_HOME;
if (!runHome || !path.isAbsolute(runHome)) throw new Error('An absolute isolated Qwen home is required');
os.homedir = () => runHome;
const blocked = new Set();
function addAncestors(directory) {
  for (let current = path.resolve(directory);;) {
    blocked.add(path.join(current, '.env'));
    blocked.add(path.join(current, '.qwen', '.env'));
    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }
}
addAncestors(process.cwd());
try { addAncestors(fs.realpathSync(process.cwd())); } catch {}
addAncestors(runHome);
if (process.env.QWEN_HOME) blocked.add(path.resolve(process.env.QWEN_HOME, '.env'));
function hidden(file) {
  if (file instanceof URL) file = file.protocol === 'file:' ? fileURLToPath(file) : null;
  if (Buffer.isBuffer(file)) file = file.toString();
  return typeof file === 'string' && blocked.has(path.resolve(file));
}
const exists = fs.existsSync;
const read = fs.readFileSync;
fs.existsSync = function(file) { return hidden(file) ? false : exists.call(fs, file); };
fs.readFileSync = function(file, ...options) {
  if (hidden(file)) {
    const error = new Error('ENOENT: run startup .env discovery is disabled');
    error.code = 'ENOENT'; error.errno = -2; error.syscall = 'open';
    throw error;
  }
  return read.call(fs, file, ...options);
};
syncBuiltinESMExports();
