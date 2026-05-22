import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PUBLIC_DIR = path.resolve(__dirname, "..", "..", "..", "public");
const OUTPUT_DIR = path.join(PUBLIC_DIR, "bundle.parts.index");
const FINGERPRINT_DIR = path.join(OUTPUT_DIR, "fingerprints");
const DEFAULT_CURRENT_FILE = path.join(FINGERPRINT_DIR, "fingerprints.records.json");
const DIFF_DIR = path.join(OUTPUT_DIR, "diff");
const DEFAULT_PREVIOUS_FILE = process.env.BUNDLE_PREVIOUS_FINGERPRINTS || "";

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

function parseArgs(argv) {
	const options = {
		previous: DEFAULT_PREVIOUS_FILE,
		current: DEFAULT_CURRENT_FILE,
		label: "default"
	};

	for (let index = 0; index < argv.length; index += 1) {
		const arg = argv[index];
		const next = argv[index + 1];

		if (arg === "--previous" && next) {
			options.previous = next;
			index += 1;
			continue;
		}

		if (arg === "--current" && next) {
			options.current = next;
			index += 1;
			continue;
		}

		if (arg === "--label" && next) {
			options.label = next;
			index += 1;
		}
	}

	return options;
}

function toMap(records) {
	const map = Object.create(null);
	for (const record of records) {
		map[record.id] = record;
	}
	return map;
}

function summarizeRecord(record) {
	return {
		id: record.id,
		parent_chunk_id: record.parent_chunk_id,
		file: record.file,
		category: record.category,
		start: record.start,
		end: record.end,
		size_bytes: record.size_bytes,
		content_sha256: record.content_sha256,
		normalized_content_sha256: record.normalized_content_sha256,
		structure_sha256: record.structure_sha256,
		record_fingerprint_sha256: record.record_fingerprint_sha256
	};
}

function compareRecords(previous, current) {
	const changes = [];

	if (previous.content_sha256 !== current.content_sha256) {
		changes.push("content_sha256");
	}

	if (previous.normalized_content_sha256 !== current.normalized_content_sha256) {
		changes.push("normalized_content_sha256");
	}

	if (previous.structure_sha256 !== current.structure_sha256) {
		changes.push("structure_sha256");
	}

	if (previous.record_fingerprint_sha256 !== current.record_fingerprint_sha256) {
		changes.push("record_fingerprint_sha256");
	}

	if (previous.size_bytes !== current.size_bytes) {
		changes.push("size_bytes");
	}

	if (previous.start !== current.start || previous.end !== current.end) {
		changes.push("range");
	}

	if (previous.parent_chunk_id !== current.parent_chunk_id) {
		changes.push("parent_chunk_id");
	}

	if (previous.file !== current.file) {
		changes.push("file");
	}

	if (previous.category !== current.category) {
		changes.push("category");
	}

	return changes;
}

function main() {
	const options = parseArgs(process.argv.slice(2));
	if (!options.previous) {
		throw new Error("Missing previous fingerprint file. Use --previous <path>.");
	}

	assertExists(options.previous, "Previous fingerprint file");
	assertExists(options.current, "Current fingerprint file");
	ensureDir(DIFF_DIR);

	const previousRecords = readJson(options.previous);
	const currentRecords = readJson(options.current);
	const previousMap = toMap(previousRecords);
	const currentMap = toMap(currentRecords);

	const previousIds = new Set(previousRecords.map((record) => record.id));
	const currentIds = new Set(currentRecords.map((record) => record.id));

	const added = [];
	const removed = [];
	const changed = [];
	const unchanged = [];

	for (const record of currentRecords) {
		if (!previousMap[record.id]) {
			added.push(summarizeRecord(record));
			continue;
		}

		const prior = previousMap[record.id];
		const fields = compareRecords(prior, record);
		if (fields.length === 0) {
			unchanged.push({
				id: record.id,
				parent_chunk_id: record.parent_chunk_id,
				file: record.file
			});
			continue;
		}

		changed.push({
			id: record.id,
			change_fields: fields,
			previous: summarizeRecord(prior),
			current: summarizeRecord(record)
		});
	}

	for (const record of previousRecords) {
		if (!currentMap[record.id]) {
			removed.push(summarizeRecord(record));
		}
	}

	const label = options.label.replace(/[^A-Za-z0-9._-]+/g, "_");
	const detailFile = path.join(DIFF_DIR, `fingerprint-diff.${label}.json`);
	const summaryFile = path.join(DIFF_DIR, `fingerprint-diff.${label}.summary.json`);

	const detail = {
		label,
		previous_file: path.resolve(options.previous),
		current_file: path.resolve(options.current),
		added,
		removed,
		changed,
		unchanged
	};

	const summary = {
		label,
		previous_file: path.resolve(options.previous),
		current_file: path.resolve(options.current),
		previous_record_count: previousIds.size,
		current_record_count: currentIds.size,
		added_count: added.length,
		removed_count: removed.length,
		changed_count: changed.length,
		unchanged_count: unchanged.length,
		changed_field_counts: changed.reduce((accumulator, item) => {
			for (const field of item.change_fields) {
				accumulator[field] = (accumulator[field] || 0) + 1;
			}
			return accumulator;
		}, {}),
		sample_added_ids: added.slice(0, 20).map((item) => item.id),
		sample_removed_ids: removed.slice(0, 20).map((item) => item.id),
		sample_changed_ids: changed.slice(0, 20).map((item) => item.id)
	};

	fs.writeFileSync(detailFile, `${JSON.stringify(detail, null, 2)}\n`, "utf8");
	fs.writeFileSync(summaryFile, `${JSON.stringify(summary, null, 2)}\n`, "utf8");

	console.log(
		JSON.stringify(
			{
				diff_dir: path.relative(PUBLIC_DIR, DIFF_DIR),
				detail_file: path.relative(PUBLIC_DIR, detailFile),
				summary_file: path.relative(PUBLIC_DIR, summaryFile),
				added_count: added.length,
				removed_count: removed.length,
				changed_count: changed.length,
				unchanged_count: unchanged.length
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
