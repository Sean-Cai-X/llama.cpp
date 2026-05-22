import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PUBLIC_DIR = path.resolve(__dirname, "..", "..", "..", "public");
const OUTPUT_DIR = path.join(PUBLIC_DIR, "bundle.parts.index");
const CHUNK_INDEX_FILE = path.join(OUTPUT_DIR, "index.chunks.json");
const REFINED_CHUNK_DIR = path.join(OUTPUT_DIR, "refined-chunks");
const REFINED_INDEX_FILE = path.join(OUTPUT_DIR, "index.refined.chunks.json");
const REFINED_REPORT_FILE = path.join(OUTPUT_DIR, "refine-report.json");

const TARGET_REFINED_CHUNK_BYTES = Number(process.env.TARGET_REFINED_CHUNK_BYTES || 90000);
const MAX_REFINED_CHUNK_BYTES = Number(process.env.MAX_REFINED_CHUNK_BYTES || 140000);

function sha256(text) {
	return crypto.createHash("sha256").update(text).digest("hex");
}

function sizeOf(text) {
	return Buffer.byteLength(text, "utf8");
}

function readJson(filePath) {
	return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function assertExists(filePath, label) {
	if (!fs.existsSync(filePath)) {
		throw new Error(`${label} not found: ${filePath}`);
	}
}

function ensureDir(dirPath) {
	fs.mkdirSync(dirPath, { recursive: true });
}

function getTopFrame(stack) {
	return stack[stack.length - 1];
}

function scanSafeBoundaries(source) {
	const stack = [{ type: "code" }];
	const boundaries = [];
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

	for (let index = 0; index < source.length; index += 1) {
		const frame = getTopFrame(stack);
		const ch = source[index];
		const next = index + 1 < source.length ? source[index + 1] : "";

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
					index += 1;
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
					index += 1;
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
					index += 1;
				} else if (ch === "/" && next === "*") {
					stack.push({ type: "blockComment" });
					index += 1;
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

		if (stack.length === 1 && getTopFrame(stack).type === "code") {
			if (ch === "\n" || ch === "\r" || ch === ";" || ch === "}") {
				boundaries.push(index + 1);
			}
		}
	}

	if (stack.length !== 1 || getTopFrame(stack).type !== "code") {
		throw new Error("Chunk ended while scanner was inside a string, regex, or comment.");
	}

	if (!boundaries.includes(source.length)) {
		boundaries.push(source.length);
	}

	return boundaries;
}

function splitOversizedSource(source) {
	const boundaries = scanSafeBoundaries(source);
	const segments = [];
	let start = 0;
	let lastBoundary = 0;

	for (const boundary of boundaries) {
		const candidateSize = sizeOf(source.slice(start, boundary));
		if (candidateSize < TARGET_REFINED_CHUNK_BYTES) {
			lastBoundary = boundary;
			continue;
		}

		if (lastBoundary > start) {
			segments.push({ start, end: lastBoundary });
			start = lastBoundary;
		} else if (candidateSize > MAX_REFINED_CHUNK_BYTES) {
			segments.push({ start, end: boundary });
			start = boundary;
		}

		lastBoundary = boundary;
	}

	if (start < source.length) {
		segments.push({ start, end: source.length });
	}

	return segments;
}

function relativeChunkPath(filePath) {
	return path.relative(PUBLIC_DIR, filePath);
}

function main() {
	assertExists(CHUNK_INDEX_FILE, "Chunk index");
	ensureDir(REFINED_CHUNK_DIR);

	const chunkIndex = readJson(CHUNK_INDEX_FILE);
	const refinedChunks = [];

	for (const chunk of chunkIndex) {
		const chunkPath = path.join(PUBLIC_DIR, chunk.file);
		assertExists(chunkPath, "Chunk file");

		const source = fs.readFileSync(chunkPath, "utf8");
		if (!chunk.oversized) {
			const passthroughPath = path.join(REFINED_CHUNK_DIR, `${chunk.chunk_id}.js`);
			fs.writeFileSync(passthroughPath, source, "utf8");

			refinedChunks.push({
				refined_chunk_id: chunk.chunk_id,
				parent_chunk_id: chunk.chunk_id,
				file: relativeChunkPath(passthroughPath),
				start: chunk.start,
				end: chunk.end,
				size_bytes: chunk.size_bytes,
				tags: chunk.tags,
				symbols: chunk.symbols,
				oversized_parent: false,
				sub_index: 0,
				sub_count: 1,
				sha256: sha256(source)
			});
			continue;
		}

		const segments = splitOversizedSource(source);
		for (let index = 0; index < segments.length; index += 1) {
			const segment = segments[index];
			const segmentSource = source.slice(segment.start, segment.end);
			const refinedChunkId = `${chunk.chunk_id}.part.${String(index).padStart(3, "0")}`;
			const refinedPath = path.join(REFINED_CHUNK_DIR, `${refinedChunkId}.js`);
			fs.writeFileSync(refinedPath, segmentSource, "utf8");

			refinedChunks.push({
				refined_chunk_id: refinedChunkId,
				parent_chunk_id: chunk.chunk_id,
				file: relativeChunkPath(refinedPath),
				start: chunk.start + segment.start,
				end: chunk.start + segment.end,
				size_bytes: sizeOf(segmentSource),
				tags: chunk.tags,
				symbols: chunk.symbols,
				oversized_parent: true,
				sub_index: index,
				sub_count: segments.length,
				sha256: sha256(segmentSource)
			});
		}
	}

	const report = {
		input_chunk_count: chunkIndex.length,
		refined_chunk_count: refinedChunks.length,
		oversized_parent_count: chunkIndex.filter((chunk) => chunk.oversized).length,
		target_refined_chunk_bytes: TARGET_REFINED_CHUNK_BYTES,
		max_refined_chunk_bytes: MAX_REFINED_CHUNK_BYTES,
		max_output_size_bytes: refinedChunks.length
			? Math.max(...refinedChunks.map((chunk) => chunk.size_bytes))
			: 0,
		min_output_size_bytes: refinedChunks.length
			? Math.min(...refinedChunks.map((chunk) => chunk.size_bytes))
			: 0,
		avg_output_size_bytes: refinedChunks.length
			? Math.round(
					refinedChunks.reduce((sum, chunk) => sum + chunk.size_bytes, 0) / refinedChunks.length
			  )
			: 0
	};

	fs.writeFileSync(REFINED_INDEX_FILE, `${JSON.stringify(refinedChunks, null, 2)}\n`, "utf8");
	fs.writeFileSync(REFINED_REPORT_FILE, `${JSON.stringify(report, null, 2)}\n`, "utf8");

	console.log(
		JSON.stringify(
			{
				refined_chunk_dir: relativeChunkPath(REFINED_CHUNK_DIR),
				refined_index_file: relativeChunkPath(REFINED_INDEX_FILE),
				report_file: relativeChunkPath(REFINED_REPORT_FILE),
				refined_chunk_count: refinedChunks.length,
				oversized_parent_count: report.oversized_parent_count
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
