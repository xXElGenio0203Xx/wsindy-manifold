const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const mj = require('mathjax-node');

mj.start();

const README = 'README.md';
const OUT_README = 'README.rendered.md';
const OUT_DIR = path.join('assets', 'readme');

function sha8(s) {
  return crypto.createHash('sha1').update(s).digest('hex').slice(0, 8);
}

function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function typesetSVG(tex) {
  return new Promise((resolve, reject) => {
    mj.typeset({ math: tex, format: 'TeX', svg: true }, (data) => {
      if (data.errors) return reject(data.errors);
      resolve(data.svg);
    });
  });
}

async function renderAll(text) {
  ensureDir(OUT_DIR);

  // First handle display math $$...$$ (multiline)
  text = await replaceAsync(text, /\$\$([\s\S]+?)\$\$/g, async (match, tex) => {
    const hash = sha8(tex);
    const filename = `eq-${hash}.svg`;
    const outPath = path.join(OUT_DIR, filename);
    if (!fs.existsSync(outPath)) {
      const svg = await typesetSVG(tex);
      fs.writeFileSync(outPath, svg, 'utf8');
      console.log('Wrote', outPath);
    }
    return `![](${outPath})`;
  });

  // Then handle inline math $...$ (avoid matching $$ which is already handled)
  text = await replaceAsync(text, /\$(?!\s)([^\n$]+?)(?!\s)\$/g, async (match, tex) => {
    // skip if looks like a URL or code fragment (heuristic)
    const hash = sha8(tex);
    const filename = `eq-${hash}.svg`;
    const outPath = path.join(OUT_DIR, filename);
    if (!fs.existsSync(outPath)) {
      const svg = await typesetSVG(tex);
      fs.writeFileSync(outPath, svg, 'utf8');
      console.log('Wrote', outPath);
    }
    return `![](${outPath})`;
  });

  return text;
}

// helper to allow async replacement
function replaceAsync(str, regex, asyncFn) {
  const matches = [];
  str.replace(regex, (match, ...args) => {
    const offset = args[args.length - 2];
    matches.push({ match, args: args.slice(0, -2), offset });
    return match;
  });
  if (matches.length === 0) return Promise.resolve(str);
  const parts = [];
  let lastIndex = 0;
  const promises = matches.map(async (m) => {
    // reconstruct the captured groups as the asyncFn expects (match, group1, group2...)
    const groups = m.args;
    const replacement = await asyncFn(m.match, ...groups);
    return { offset: m.offset, match: m.match, replacement };
  });
  return Promise.all(promises).then((repls) => {
    // build final string
    let out = '';
    let cursor = 0;
    let idx = 0;
    str.replace(regex, (match) => {
      const r = repls[idx++];
      const pos = str.indexOf(match, cursor);
      out += str.slice(cursor, pos) + r.replacement;
      cursor = pos + match.length;
      return match;
    });
    out += str.slice(cursor);
    return out;
  });
}

async function main() {
  if (!fs.existsSync(README)) {
    console.error('No README.md found in repository root.');
    process.exit(1);
  }
  const input = fs.readFileSync(README, 'utf8');
  const output = await renderAll(input);
  fs.writeFileSync(OUT_README, output, 'utf8');
  console.log('Wrote', OUT_README);
}

main().catch((err) => {
  console.error('Error rendering README math:', err);
  process.exit(2);
});
