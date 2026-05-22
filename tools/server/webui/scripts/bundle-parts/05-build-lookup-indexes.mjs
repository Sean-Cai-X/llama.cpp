import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PUBLIC_DIR = path.resolve(__dirname, "..", "..", "..", "public");
const OUTPUT_DIR = path.join(PUBLIC_DIR, "bundle.parts.index");
const REFINED_INDEX_FILE = path.join(OUTPUT_DIR, "index.refined.chunks.json");
const LOOKUP_DIR = path.join(OUTPUT_DIR, "lookup");
const CHUNKS_BY_TAG_FILE = path.join(LOOKUP_DIR, "chunks.by-tag.json");
const CHUNKS_BY_SYMBOL_FILE = path.join(LOOKUP_DIR, "chunks.by-symbol.json");
const CHUNKS_BY_PARENT_FILE = path.join(LOOKUP_DIR, "chunks.by-parent.json");
const CHUNKS_BY_FILE_FILE = path.join(LOOKUP_DIR, "chunks.by-file.json");
const LOOKUP_SUMMARY_FILE = path.join(LOOKUP_DIR, "lookup-summary.json");

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

function unique(items) {
	return [...new Set(items.filter(Boolean))];
}

function addGroupedRecord(grouped, key, value) {
	if (!key) {
		return;
	}

	if (!Array.isArray(grouped[key])) {
		grouped[key] = [];
	}

	grouped[key].push(value);
}

function buildChunkStub(chunk) {
	return {
		refined_chunk_id: chunk.refined_chunk_id,
		parent_chunk_id: chunk.parent_chunk_id,
		file: chunk.file,
		start: chunk.start,
		end: chunk.end,
		size_bytes: chunk.size_bytes,
		oversized_parent: !!chunk.oversized_parent,
		sub_index: chunk.sub_index,
		sub_count: chunk.sub_count
	};
}

function finalizeGrouped(grouped, keyName) {
	return Object.entries(grouped)
		.sort(([leftKey], [rightKey]) => leftKey.localeCompare(rightKey))
		.map(([key, chunks]) => ({
			[keyName]: key,
			count: chunks.length,
			chunks: chunks.sort((left, right) => {
				if (left.parent_chunk_id !== right.parent_chunk_id) {
					return left.parent_chunk_id.localeCompare(right.parent_chunk_id);
				}

				return left.start - right.start;
			})
		}));
}

function main() {
	assertExists(REFINED_INDEX_FILE, "Refined chunk index");
	ensureDir(LOOKUP_DIR);

	const refinedChunks = readJson(REFINED_INDEX_FILE);
	const byTag = Object.create(null);
	const bySymbol = Object.create(null);
	const byParent = Object.create(null);
	const byFile = Object.create(null);

	for (const chunk of refinedChunks) {
		const stub = buildChunkStub(chunk);

		addGroupedRecord(byParent, chunk.parent_chunk_id, stub);
		addGroupedRecord(byFile, chunk.file, stub);

		for (const tag of unique(chunk.tags || [])) {
			addGroupedRecord(byTag, tag, stub);
		}

		for (const symbol of unique(chunk.symbols || [])) {
			addGroupedRecord(bySymbol, symbol, stub);
		}
	}

	const groupedByTag = finalizeGrouped(byTag, "tag");
	const groupedBySymbol = finalizeGrouped(bySymbol, "symbol");
	const groupedByParent = finalizeGrouped(byParent, "parent_chunk_id");
	const groupedByFile = finalizeGrouped(byFile, "file");

	const summary = {
		refined_chunk_count: refinedChunks.length,
		tag_group_count: groupedByTag.length,
		symbol_group_count: groupedBySymbol.length,
		parent_group_count: groupedByParent.length,
		file_group_count: groupedByFile.length,
		largest_symbol_groups: [...groupedBySymbol]
			.sort((left, right) => right.count - left.count)
			.slice(0, 30)
			.map((entry) => ({
				symbol: entry.symbol,
				count: entry.count
			})),
		largest_tag_groups: [...groupedByTag]
			.sort((left, right) => right.count - left.count)
			.slice(0, 30)
			.map((entry) => ({
				tag: entry.tag,
				count: entry.count
			}))
	};

	fs.writeFileSync(CHUNKS_BY_TAG_FILE, `${JSON.stringify(groupedByTag, null, 2)}\n`, "utf8");
	fs.writeFileSync(CHUNKS_BY_SYMBOL_FILE, `${JSON.stringify(groupedBySymbol, null, 2)}\n`, "utf8");
	fs.writeFileSync(CHUNKS_BY_PARENT_FILE, `${JSON.stringify(groupedByParent, null, 2)}\n`, "utf8");
	fs.writeFileSync(CHUNKS_BY_FILE_FILE, `${JSON.stringify(groupedByFile, null, 2)}\n`, "utf8");
	fs.writeFileSync(LOOKUP_SUMMARY_FILE, `${JSON.stringify(summary, null, 2)}\n`, "utf8");

	console.log(
		JSON.stringify(
			{
				lookup_dir: path.relative(PUBLIC_DIR, LOOKUP_DIR),
				by_tag_file: path.relative(PUBLIC_DIR, CHUNKS_BY_TAG_FILE),
				by_symbol_file: path.relative(PUBLIC_DIR, CHUNKS_BY_SYMBOL_FILE),
				by_parent_file: path.relative(PUBLIC_DIR, CHUNKS_BY_PARENT_FILE),
				by_file_file: path.relative(PUBLIC_DIR, CHUNKS_BY_FILE_FILE),
				summary_file: path.relative(PUBLIC_DIR, LOOKUP_SUMMARY_FILE),
				refined_chunk_count: refinedChunks.length
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
