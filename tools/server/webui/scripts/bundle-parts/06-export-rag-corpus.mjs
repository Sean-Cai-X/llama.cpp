import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PUBLIC_DIR = path.resolve(__dirname, "..", "..", "..", "public");
const OUTPUT_DIR = path.join(PUBLIC_DIR, "bundle.parts.index");
const REFINED_INDEX_FILE = path.join(OUTPUT_DIR, "index.refined.chunks.json");
const LOOKUP_SUMMARY_FILE = path.join(OUTPUT_DIR, "lookup", "lookup-summary.json");
const CORPUS_DIR = path.join(OUTPUT_DIR, "corpus");
const JSONL_FILE = path.join(CORPUS_DIR, "rag-corpus.jsonl");
const MANIFEST_FILE = path.join(CORPUS_DIR, "rag-corpus.manifest.json");

function assertExists(filePath, label) {
	if (!fs.existsSync(filePath)) {
		throw new Error(`${label} not found: ${filePath}`);
	}
}

function ensureDir(dirPath) {
	fs.mkdirSync(dirPath, { recursive: true });
}

function readJson(filePath) {
	return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function readText(filePath) {
	return fs.readFileSync(filePath, "utf8");
}

function normalizeText(text) {
	return text.replace(/\r\n/g, "\n");
}

function summarizeText(text) {
	const normalized = normalizeText(text);
	const nonEmptyLines = normalized
		.split("\n")
		.map((line) => line.trim())
		.filter(Boolean);

	return nonEmptyLines.slice(0, 3).join(" ").slice(0, 240);
}

function unique(items) {
	return [...new Set((items || []).filter(Boolean))];
}

function classifyChunk(chunk) {
	if (chunk.tags?.includes("mcp")) {
		return "mcp";
	}

	if (chunk.tags?.includes("chat-ui")) {
		return "chat-ui";
	}

	if (chunk.tags?.includes("svelte-compiled")) {
		return "svelte-compiled";
	}

	if (chunk.tags?.length) {
		return chunk.tags[0];
	}

	return "generic-js";
}

function buildCorpusRecord(chunk, content) {
	return {
		id: chunk.refined_chunk_id,
		parent_chunk_id: chunk.parent_chunk_id,
		file: chunk.file,
		source_kind: "bundle.parts.refined_chunk",
		category: classifyChunk(chunk),
		start: chunk.start,
		end: chunk.end,
		size_bytes: chunk.size_bytes,
		oversized_parent: !!chunk.oversized_parent,
		sub_index: chunk.sub_index,
		sub_count: chunk.sub_count,
		tags: unique(chunk.tags),
		symbols: unique(chunk.symbols),
		summary: summarizeText(content),
		content
	};
}

function main() {
	assertExists(REFINED_INDEX_FILE, "Refined chunk index");
	ensureDir(CORPUS_DIR);

	const refinedChunks = readJson(REFINED_INDEX_FILE);
	const lookupSummary = fs.existsSync(LOOKUP_SUMMARY_FILE) ? readJson(LOOKUP_SUMMARY_FILE) : null;
	const records = [];

	for (const chunk of refinedChunks) {
		const filePath = path.join(PUBLIC_DIR, chunk.file);
		assertExists(filePath, "Refined chunk file");
		const content = normalizeText(readText(filePath));
		records.push(buildCorpusRecord(chunk, content));
	}

	const jsonl = records.map((record) => JSON.stringify(record)).join("\n");
	fs.writeFileSync(JSONL_FILE, `${jsonl}\n`, "utf8");

	const manifest = {
		record_count: records.length,
		output_file: path.relative(PUBLIC_DIR, JSONL_FILE),
		lookup_summary_file: lookupSummary ? path.relative(PUBLIC_DIR, LOOKUP_SUMMARY_FILE) : null,
		category_counts: records.reduce((accumulator, record) => {
			accumulator[record.category] = (accumulator[record.category] || 0) + 1;
			return accumulator;
		}, {}),
		tag_counts: records.reduce((accumulator, record) => {
			for (const tag of record.tags) {
				accumulator[tag] = (accumulator[tag] || 0) + 1;
			}
			return accumulator;
		}, {}),
		largest_records: [...records]
			.sort((left, right) => right.size_bytes - left.size_bytes)
			.slice(0, 30)
			.map((record) => ({
				id: record.id,
				parent_chunk_id: record.parent_chunk_id,
				file: record.file,
				size_bytes: record.size_bytes,
				category: record.category,
				tags: record.tags
			}))
	};

	fs.writeFileSync(MANIFEST_FILE, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

	console.log(
		JSON.stringify(
			{
				corpus_dir: path.relative(PUBLIC_DIR, CORPUS_DIR),
				jsonl_file: path.relative(PUBLIC_DIR, JSONL_FILE),
				manifest_file: path.relative(PUBLIC_DIR, MANIFEST_FILE),
				record_count: records.length
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
