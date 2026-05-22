import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PUBLIC_DIR = path.resolve(__dirname, "..", "..", "..", "public");
const OUTPUT_DIR = path.join(PUBLIC_DIR, "bundle.parts.index");
const DIFF_DIR = path.join(OUTPUT_DIR, "diff");

function assertExists(filePath, label) {
	if (!fs.existsSync(filePath)) {
		throw new Error(`${label} not found: ${filePath}`);
	}
}

function readJson(filePath) {
	return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function parseArgs(argv) {
	const options = {
		input: "",
		label: ""
	};

	for (let index = 0; index < argv.length; index += 1) {
		const arg = argv[index];
		const next = argv[index + 1];

		if (arg === "--input" && next) {
			options.input = next;
			index += 1;
			continue;
		}

		if (arg === "--label" && next) {
			options.label = next;
			index += 1;
		}
	}

	if (!options.input) {
		throw new Error("Missing --input <diff-json-path>.");
	}

	return options;
}

function incrementCounter(map, key, amount = 1) {
	if (!key) {
		return;
	}

	map[key] = (map[key] || 0) + amount;
}

function sortCounterEntries(counter, keyName) {
	return Object.entries(counter)
		.sort((left, right) => {
			if (right[1] !== left[1]) {
				return right[1] - left[1];
			}

			return left[0].localeCompare(right[0]);
		})
		.map(([key, count]) => ({
			[keyName]: key,
			count
		}));
}

function shortRecord(record) {
	if (!record) {
		return null;
	}

	return {
		id: record.id,
		parent_chunk_id: record.parent_chunk_id,
		file: record.file,
		category: record.category,
		start: record.start,
		end: record.end,
		size_bytes: record.size_bytes
	};
}

function main() {
	const options = parseArgs(process.argv.slice(2));
	assertExists(options.input, "Diff detail file");

	const detail = readJson(options.input);
	const label = (options.label || detail.label || "default").replace(/[^A-Za-z0-9._-]+/g, "_");
	const summaryReportFile = path.join(DIFF_DIR, `fingerprint-diff.${label}.report.json`);

	const addedByCategory = Object.create(null);
	const removedByCategory = Object.create(null);
	const changedByCategory = Object.create(null);
	const changedByField = Object.create(null);
	const changedByParent = Object.create(null);

	for (const item of detail.added || []) {
		incrementCounter(addedByCategory, item.category);
	}

	for (const item of detail.removed || []) {
		incrementCounter(removedByCategory, item.category);
	}

	for (const item of detail.changed || []) {
		const current = item.current || {};
		incrementCounter(changedByCategory, current.category || item.previous?.category);
		incrementCounter(changedByParent, current.parent_chunk_id || item.previous?.parent_chunk_id);

		for (const field of item.change_fields || []) {
			incrementCounter(changedByField, field);
		}
	}

	const report = {
		label,
		source_diff_file: path.resolve(options.input),
		added_count: (detail.added || []).length,
		removed_count: (detail.removed || []).length,
		changed_count: (detail.changed || []).length,
		unchanged_count: (detail.unchanged || []).length,
		added_by_category: sortCounterEntries(addedByCategory, "category"),
		removed_by_category: sortCounterEntries(removedByCategory, "category"),
		changed_by_category: sortCounterEntries(changedByCategory, "category"),
		changed_by_field: sortCounterEntries(changedByField, "field"),
		changed_by_parent_chunk: sortCounterEntries(changedByParent, "parent_chunk_id"),
		sample_added: (detail.added || []).slice(0, 20).map((item) => shortRecord(item)),
		sample_removed: (detail.removed || []).slice(0, 20).map((item) => shortRecord(item)),
		sample_changed: (detail.changed || []).slice(0, 20).map((item) => ({
			id: item.id,
			change_fields: item.change_fields || [],
			previous: shortRecord(item.previous),
			current: shortRecord(item.current)
		}))
	};

	fs.writeFileSync(summaryReportFile, `${JSON.stringify(report, null, 2)}\n`, "utf8");

	console.log(
		JSON.stringify(
			{
				diff_dir: path.relative(PUBLIC_DIR, DIFF_DIR),
				report_file: path.relative(PUBLIC_DIR, summaryReportFile),
				added_count: report.added_count,
				removed_count: report.removed_count,
				changed_count: report.changed_count,
				unchanged_count: report.unchanged_count
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
