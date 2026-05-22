import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PUBLIC_DIR = path.resolve(__dirname, "..", "..", "..", "public");
const OUTPUT_DIR = path.join(PUBLIC_DIR, "bundle.parts.index");
const INPUT_BUNDLE = path.join(OUTPUT_DIR, "rebuilt.bundle.js");
const STRUCTURE_INDEX_FILE = path.join(OUTPUT_DIR, "index.structure.raw.json");
const SYMBOL_INDEX_FILE = path.join(OUTPUT_DIR, "index.symbols.raw.json");
const CHUNK_DIR = path.join(OUTPUT_DIR, "chunks");
const CHUNK_INDEX_FILE = path.join(OUTPUT_DIR, "index.chunks.json");
const SYMBOLS_BY_CHUNK_FILE = path.join(OUTPUT_DIR, "index.symbols.by_chunk.json");
const REPORT_FILE = path.join(OUTPUT_DIR, "slice-report.json");

const TARGET_CHUNK_BYTES = Number(process.env.TARGET_CHUNK_BYTES || 120000);
const MAX_CHUNK_BYTES = Number(process.env.MAX_CHUNK_BYTES || 220000);

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

function unique(items) {
	return [...new Set(items.filter(Boolean))];
}

function overlaps(leftStart, leftEnd, rightStart, rightEnd) {
	return leftStart < rightEnd && rightStart < leftEnd;
}

function getLineRange(nodes) {
	const starts = nodes.map((node) => node.line_start).filter(Number.isFinite);
	const ends = nodes.map((node) => node.line_end).filter(Number.isFinite);

	return {
		line_start: starts.length ? Math.min(...starts) : null,
		line_end: ends.length ? Math.max(...ends) : null
	};
}

function makeChunkId(index) {
	return `chunk_${String(index).padStart(6, "0")}`;
}

function writeChunk(chunkId, source) {
	const filePath = path.join(CHUNK_DIR, `${chunkId}.js`);
	fs.writeFileSync(filePath, source, "utf8");
	return path.relative(PUBLIC_DIR, filePath);
}

function createChunkRecord({ chunkId, source, start, end, nodes, oversized, reason }) {
	const lines = getLineRange(nodes);

	return {
		chunk_id: chunkId,
		file: null,
		kind: oversized ? "oversized_top_level_node" : "top_level_group",
		reason: reason || null,
		start,
		end,
		size_bytes: sizeOf(source),
		line_start: lines.line_start,
		line_end: lines.line_end,
		top_level_ids: nodes.map((node) => node.id),
		top_level_count: nodes.length,
		top_level_kinds: unique(nodes.map((node) => node.kind)),
		tags: unique(nodes.flatMap((node) => node.tags || [])),
		symbols: unique(nodes.flatMap((node) => node.symbols || [])),
		oversized: !!oversized,
		sha256: sha256(source)
	};
}

function collectChunkSymbols(chunk, symbolIndex) {
	return symbolIndex
		.filter((symbol) => overlaps(chunk.start, chunk.end, symbol.start, symbol.end))
		.map((symbol) => ({
			symbol: symbol.symbol,
			kind: symbol.kind,
			start: symbol.start,
			end: symbol.end,
			top_level_id: symbol.top_level_id,
			tags: symbol.tags || []
		}));
}

function main() {
	assertExists(INPUT_BUNDLE, "Rebuilt bundle");
	assertExists(STRUCTURE_INDEX_FILE, "Structure index");
	assertExists(SYMBOL_INDEX_FILE, "Symbol index");
	ensureDir(CHUNK_DIR);

	const code = fs.readFileSync(INPUT_BUNDLE, "utf8");
	const structureIndex = readJson(STRUCTURE_INDEX_FILE).sort((left, right) => left.order - right.order);
	const symbolIndex = readJson(SYMBOL_INDEX_FILE);

	const chunks = [];
	let currentNodes = [];
	let currentStart = null;
	let currentEnd = null;

	function flushCurrent() {
		if (!currentNodes.length) {
			return;
		}

		const source = code.slice(currentStart, currentEnd);
		const chunkId = makeChunkId(chunks.length);
		const chunk = createChunkRecord({
			chunkId,
			source,
			start: currentStart,
			end: currentEnd,
			nodes: currentNodes,
			oversized: false,
			reason: null
		});

		chunk.file = writeChunk(chunkId, source);
		chunks.push(chunk);

		currentNodes = [];
		currentStart = null;
		currentEnd = null;
	}

	for (const node of structureIndex) {
		const nodeSource = code.slice(node.start, node.end);
		const nodeSize = sizeOf(nodeSource);

		if (nodeSize > MAX_CHUNK_BYTES) {
			flushCurrent();

			const chunkId = makeChunkId(chunks.length);
			const chunk = createChunkRecord({
				chunkId,
				source: nodeSource,
				start: node.start,
				end: node.end,
				nodes: [node],
				oversized: true,
				reason: `single_top_level_node_exceeds_max_${MAX_CHUNK_BYTES}_bytes`
			});

			chunk.file = writeChunk(chunkId, nodeSource);
			chunks.push(chunk);
			continue;
		}

		if (!currentNodes.length) {
			currentNodes = [node];
			currentStart = node.start;
			currentEnd = node.end;
			continue;
		}

		const mergedSource = code.slice(currentStart, node.end);
		const mergedSize = sizeOf(mergedSource);
		if (mergedSize > TARGET_CHUNK_BYTES || mergedSize > MAX_CHUNK_BYTES) {
			flushCurrent();
			currentNodes = [node];
			currentStart = node.start;
			currentEnd = node.end;
			continue;
		}

		currentNodes.push(node);
		currentEnd = node.end;
	}

	flushCurrent();

	const symbolsByChunk = chunks.map((chunk) => ({
		chunk_id: chunk.chunk_id,
		file: chunk.file,
		symbols: collectChunkSymbols(chunk, symbolIndex)
	}));

	const report = {
		input_file: path.relative(PUBLIC_DIR, INPUT_BUNDLE),
		input_size_bytes: sizeOf(code),
		target_chunk_size_bytes: TARGET_CHUNK_BYTES,
		max_chunk_size_bytes: MAX_CHUNK_BYTES,
		top_level_node_count: structureIndex.length,
		chunk_count: chunks.length,
		oversized_chunk_count: chunks.filter((chunk) => chunk.oversized).length,
		total_chunk_bytes: chunks.reduce((sum, chunk) => sum + chunk.size_bytes, 0),
		min_chunk_bytes: chunks.length ? Math.min(...chunks.map((chunk) => chunk.size_bytes)) : 0,
		max_chunk_bytes: chunks.length ? Math.max(...chunks.map((chunk) => chunk.size_bytes)) : 0,
		avg_chunk_bytes: chunks.length
			? Math.round(chunks.reduce((sum, chunk) => sum + chunk.size_bytes, 0) / chunks.length)
			: 0,
		largest_chunks: [...chunks]
			.sort((left, right) => right.size_bytes - left.size_bytes)
			.slice(0, 20)
			.map((chunk) => ({
				chunk_id: chunk.chunk_id,
				file: chunk.file,
				size_bytes: chunk.size_bytes,
				oversized: chunk.oversized,
				top_level_count: chunk.top_level_count,
				top_level_kinds: chunk.top_level_kinds,
				tags: chunk.tags,
				symbols: chunk.symbols.slice(0, 30)
			})),
		tag_counts: chunks.reduce((accumulator, chunk) => {
			for (const tag of chunk.tags) {
				accumulator[tag] = (accumulator[tag] || 0) + 1;
			}

			return accumulator;
		}, {})
	};

	fs.writeFileSync(CHUNK_INDEX_FILE, `${JSON.stringify(chunks, null, 2)}\n`, "utf8");
	fs.writeFileSync(SYMBOLS_BY_CHUNK_FILE, `${JSON.stringify(symbolsByChunk, null, 2)}\n`, "utf8");
	fs.writeFileSync(REPORT_FILE, `${JSON.stringify(report, null, 2)}\n`, "utf8");

	console.log(
		JSON.stringify(
			{
				chunk_dir: path.relative(PUBLIC_DIR, CHUNK_DIR),
				chunk_index_file: path.relative(PUBLIC_DIR, CHUNK_INDEX_FILE),
				symbol_map_file: path.relative(PUBLIC_DIR, SYMBOLS_BY_CHUNK_FILE),
				report_file: path.relative(PUBLIC_DIR, REPORT_FILE),
				chunk_count: chunks.length,
				oversized_chunk_count: report.oversized_chunk_count
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
