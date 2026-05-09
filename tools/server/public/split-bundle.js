#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const PUBLIC_DIR = __dirname;
const SOURCE_FILE = path.join(PUBLIC_DIR, "bundle.original.js");
const PARTS_DIR = path.join(PUBLIC_DIR, "bundle.parts");
const TARGET_PART_SIZE = 480000;
const MAX_PART_SIZE = 500000;
function ensureSourceExists(filePath) {
	if (!fs.existsSync(filePath)) {
		throw new Error(`Source bundle not found: ${filePath}`);
	}
}

function getTopFrame(stack) {
	return stack[stack.length - 1];
}

function scanPartBoundaries(source) {
	const stack = [{ type: "code" }];
	const parts = [];
	let chunkStart = 0;
	let lastSafeBoundary = -1;
	let allowRegexAfter = true;

	function noteExpressionToken(ch) {
		if (/\s/.test(ch)) {
			return;
		}

		if (/[A-Za-z0-9_$)\]}]/.test(ch)) {
			allowRegexAfter = false;
			return;
		}

		allowRegexAfter = true;
	}

	for (let i = 0; i < source.length; i += 1) {
		const frame = getTopFrame(stack);
		const ch = source[i];
		const next = i + 1 < source.length ? source[i + 1] : "";

		switch (frame.type) {
			case "single":
			case "double": {
				if (frame.escaped) {
					frame.escaped = false;
				} else if (ch === "\\") {
					frame.escaped = true;
				} else if (ch === frame.quote) {
					stack.pop();
				}
				break;
			}
			case "lineComment": {
				if (ch === "\n" || ch === "\r") {
					stack.pop();
				}
				break;
			}
			case "blockComment": {
				if (ch === "*" && next === "/") {
					stack.pop();
					i += 1;
				}
				break;
			}
			case "regex": {
				if (frame.escaped) {
					frame.escaped = false;
				} else if (ch === "\\") {
					frame.escaped = true;
				} else if (ch === "[" && !frame.inCharClass) {
					frame.inCharClass = true;
				} else if (ch === "]" && frame.inCharClass) {
					frame.inCharClass = false;
				} else if (ch === "/" && !frame.inCharClass) {
					stack.pop();
					allowRegexAfter = false;
				}
				break;
			}
			case "templateText": {
				if (frame.escaped) {
					frame.escaped = false;
				} else if (ch === "\\") {
					frame.escaped = true;
				} else if (ch === "`") {
					stack.pop();
				} else if (ch === "$" && next === "{") {
					stack.push({ type: "templateExpr", braceDepth: 1 });
					i += 1;
				}
				break;
			}
			case "code":
			case "templateExpr": {
				if (frame.type === "templateExpr") {
					if (ch === "{") {
						frame.braceDepth += 1;
					} else if (ch === "}") {
						frame.braceDepth -= 1;
						if (frame.braceDepth === 0) {
							stack.pop();
							break;
						}
					}
				}

				if (ch === "'") {
					stack.push({ type: "single", quote: "'", escaped: false });
					allowRegexAfter = false;
				} else if (ch === "\"") {
					stack.push({ type: "double", quote: "\"", escaped: false });
					allowRegexAfter = false;
				} else if (ch === "`") {
					stack.push({ type: "templateText", escaped: false });
					allowRegexAfter = false;
				} else if (ch === "/" && next === "/") {
					stack.push({ type: "lineComment" });
					i += 1;
				} else if (ch === "/" && next === "*") {
					stack.push({ type: "blockComment" });
					i += 1;
				} else if (ch === "/" && allowRegexAfter) {
					stack.push({ type: "regex", escaped: false, inCharClass: false });
				} else {
					noteExpressionToken(ch);
				}
				break;
			}
			default:
				throw new Error(`Unknown scanner state: ${frame.type}`);
		}

		const top = getTopFrame(stack);
		if (stack.length === 1 && top.type === "code" && (ch === "\n" || ch === "\r")) {
			lastSafeBoundary = i + 1;
		}

		const currentChunkSize = i + 1 - chunkStart;
		if (currentChunkSize >= TARGET_PART_SIZE && lastSafeBoundary > chunkStart) {
			parts.push({ start: chunkStart, end: lastSafeBoundary });
			chunkStart = lastSafeBoundary;
			lastSafeBoundary = -1;
		} else if (currentChunkSize > MAX_PART_SIZE) {
			throw new Error(
				`Unable to find a safe split boundary before ${MAX_PART_SIZE} bytes near offset ${i}.`
			);
		}
	}

	if (stack.length !== 1 || getTopFrame(stack).type !== "code") {
		throw new Error("Source ended while scanner was still inside a string/comment/template.");
	}

	if (chunkStart < source.length) {
		parts.push({ start: chunkStart, end: source.length });
	}

	return parts;
}

function writeParts(source, boundaries) {
	fs.rmSync(PARTS_DIR, { recursive: true, force: true });
	fs.mkdirSync(PARTS_DIR, { recursive: true });

	const partNames = [];
	let largestPart = 0;

	boundaries.forEach((range, index) => {
		const fileName = `bundle.part.${String(index).padStart(3, "0")}.js`;
		const filePath = path.join(PARTS_DIR, fileName);
		const content = source.slice(range.start, range.end);
		const size = Buffer.byteLength(content, "utf8");

		if (size >= MAX_PART_SIZE) {
			throw new Error(`${fileName} is ${size} bytes, which exceeds the < 500KB limit.`);
		}

		fs.writeFileSync(filePath, content, "utf8");
		partNames.push(fileName);
		if (size > largestPart) {
			largestPart = size;
		}
	});

	const manifest = {
		record_model: "bundle_parts_v1",
		source: "bundle.original.js",
		part_count: partNames.length,
		parts: partNames,
		target_part_size_bytes: TARGET_PART_SIZE,
		max_part_size_bytes: MAX_PART_SIZE,
		largest_part_size_bytes: largestPart
	};

	fs.writeFileSync(
		path.join(PARTS_DIR, "manifest.json"),
		`${JSON.stringify(manifest, null, 2)}\n`,
		"utf8"
	);

	return manifest;
}

function main() {
	ensureSourceExists(SOURCE_FILE);
	const source = fs.readFileSync(SOURCE_FILE, "utf8");
	const boundaries = scanPartBoundaries(source);
	const manifest = writeParts(source, boundaries);

	console.log(
		JSON.stringify(
			{
				source: manifest.source,
				part_count: manifest.part_count,
				largest_part_size_bytes: manifest.largest_part_size_bytes
			},
			null,
			2
		)
	);
}

try {
	main();
} catch (error) {
	console.error(error && error.stack ? error.stack : error);
	process.exitCode = 1;
}
