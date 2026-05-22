import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PUBLIC_DIR = path.resolve(__dirname, "..", "..", "..", "public");
const OUTPUT_DIR = path.join(PUBLIC_DIR, "bundle.parts.index");
const CORPUS_DIR = path.join(OUTPUT_DIR, "corpus");
const JSONL_FILE = path.join(CORPUS_DIR, "rag-corpus.jsonl");
const FINGERPRINT_DIR = path.join(OUTPUT_DIR, "fingerprints");
const RECORDS_FILE = path.join(FINGERPRINT_DIR, "fingerprints.records.json");
const BY_ID_FILE = path.join(FINGERPRINT_DIR, "fingerprints.by-id.json");
const BY_PARENT_FILE = path.join(FINGERPRINT_DIR, "fingerprints.by-parent.json");
const BY_CONTENT_HASH_FILE = path.join(FINGERPRINT_DIR, "fingerprints.by-content-hash.json");
const MANIFEST_FILE = path.join(FINGERPRINT_DIR, "fingerprints.manifest.json");

function assertExists(filePath, label) {
	if (!fs.existsSync(filePath)) {
		throw new Error(`${label} not found: ${filePath}`);
	}
}

function ensureDir(dirPath) {
	fs.mkdirSync(dirPath, { recursive: true });
}

function sha256(text) {
	return crypto.createHash("sha256").update(text).digest("hex");
}

function readJsonl(filePath) {
	return fs
		.readFileSync(filePath, "utf8")
		.split(/\r?\n/)
		.map((line) => line.trim())
		.filter(Boolean)
		.map((line) => JSON.parse(line));
}

function normalizeContent(content) {
	return String(content || "")
		.replace(/\r\n/g, "\n")
		.replace(/[ \t]+/g, " ")
		.trim();
}

function makeRecordFingerprint(record) {
	const normalizedContent = normalizeContent(record.content);
	const symbolKey = (record.symbols || []).slice().sort().join("|");
	const tagKey = (record.tags || []).slice().sort().join("|");
	const locationKey = `${record.file}:${record.start}-${record.end}`;

	return {
		id: record.id,
		parent_chunk_id: record.parent_chunk_id,
		file: record.file,
		category: record.category,
		start: record.start,
		end: record.end,
		size_bytes: record.size_bytes,
		symbol_count: (record.symbols || []).length,
		tag_count: (record.tags || []).length,
		content_sha256: sha256(record.content || ""),
		normalized_content_sha256: sha256(normalizedContent),
		structure_sha256: sha256(
			JSON.stringify({
				parent_chunk_id: record.parent_chunk_id,
				file: record.file,
				category: record.category,
				start: record.start,
				end: record.end,
				symbols: (record.symbols || []).slice().sort(),
				tags: (record.tags || []).slice().sort()
			})
		),
		record_fingerprint_sha256: sha256(
			[
				record.id,
				record.parent_chunk_id,
				locationKey,
				symbolKey,
				tagKey,
				sha256(normalizedContent)
			].join("||")
		)
	};
}

function pushGrouped(map, key, value) {
	if (!key) {
		return;
	}

	if (!Array.isArray(map[key])) {
		map[key] = [];
	}

	map[key].push(value);
}

function buildStub(entry) {
	return {
		id: entry.id,
		parent_chunk_id: entry.parent_chunk_id,
		file: entry.file,
		category: entry.category,
		start: entry.start,
		end: entry.end,
		size_bytes: entry.size_bytes,
		content_sha256: entry.content_sha256,
		normalized_content_sha256: entry.normalized_content_sha256,
		structure_sha256: entry.structure_sha256,
		record_fingerprint_sha256: entry.record_fingerprint_sha256
	};
}

function finalizeGroups(grouped, keyName) {
	return Object.entries(grouped)
		.sort(([leftKey], [rightKey]) => leftKey.localeCompare(rightKey))
		.map(([key, entries]) => ({
			[keyName]: key,
			count: entries.length,
			records: entries.sort((left, right) => {
				if (left.parent_chunk_id !== right.parent_chunk_id) {
					return left.parent_chunk_id.localeCompare(right.parent_chunk_id);
				}

				return left.start - right.start;
			})
		}));
}

function main() {
	assertExists(JSONL_FILE, "RAG corpus jsonl");
	ensureDir(FINGERPRINT_DIR);

	const records = readJsonl(JSONL_FILE);
	const fingerprints = records.map((record) => makeRecordFingerprint(record));

	const byId = Object.create(null);
	const byParent = Object.create(null);
	const byContentHash = Object.create(null);

	for (const fingerprint of fingerprints) {
		byId[fingerprint.id] = buildStub(fingerprint);
		pushGrouped(byParent, fingerprint.parent_chunk_id, buildStub(fingerprint));
		pushGrouped(byContentHash, fingerprint.normalized_content_sha256, buildStub(fingerprint));
	}

	const groupedByParent = finalizeGroups(byParent, "parent_chunk_id");
	const groupedByContentHash = finalizeGroups(byContentHash, "normalized_content_sha256");

	const duplicateContentGroups = groupedByContentHash
		.filter((entry) => entry.count > 1)
		.sort((left, right) => right.count - left.count);

	const manifest = {
		record_count: fingerprints.length,
		parent_group_count: groupedByParent.length,
		content_hash_group_count: groupedByContentHash.length,
		duplicate_content_group_count: duplicateContentGroups.length,
		duplicate_record_count: duplicateContentGroups.reduce((sum, entry) => sum + entry.count, 0),
		largest_duplicate_groups: duplicateContentGroups.slice(0, 30).map((entry) => ({
			normalized_content_sha256: entry.normalized_content_sha256,
			count: entry.count,
			record_ids: entry.records.map((record) => record.id)
		}))
	};

	fs.writeFileSync(RECORDS_FILE, `${JSON.stringify(fingerprints, null, 2)}\n`, "utf8");
	fs.writeFileSync(BY_ID_FILE, `${JSON.stringify(byId, null, 2)}\n`, "utf8");
	fs.writeFileSync(BY_PARENT_FILE, `${JSON.stringify(groupedByParent, null, 2)}\n`, "utf8");
	fs.writeFileSync(BY_CONTENT_HASH_FILE, `${JSON.stringify(groupedByContentHash, null, 2)}\n`, "utf8");
	fs.writeFileSync(MANIFEST_FILE, `${JSON.stringify(manifest, null, 2)}\n`, "utf8");

	console.log(
		JSON.stringify(
			{
				fingerprint_dir: path.relative(PUBLIC_DIR, FINGERPRINT_DIR),
				records_file: path.relative(PUBLIC_DIR, RECORDS_FILE),
				by_id_file: path.relative(PUBLIC_DIR, BY_ID_FILE),
				by_parent_file: path.relative(PUBLIC_DIR, BY_PARENT_FILE),
				by_content_hash_file: path.relative(PUBLIC_DIR, BY_CONTENT_HASH_FILE),
				manifest_file: path.relative(PUBLIC_DIR, MANIFEST_FILE),
				record_count: fingerprints.length,
				duplicate_content_group_count: manifest.duplicate_content_group_count
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
