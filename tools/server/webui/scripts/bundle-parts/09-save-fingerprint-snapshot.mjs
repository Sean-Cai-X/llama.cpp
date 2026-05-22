import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PUBLIC_DIR = path.resolve(__dirname, "..", "..", "..", "public");
const OUTPUT_DIR = path.join(PUBLIC_DIR, "bundle.parts.index");
const FINGERPRINT_DIR = path.join(OUTPUT_DIR, "fingerprints");
const SOURCE_RECORDS_FILE = path.join(FINGERPRINT_DIR, "fingerprints.records.json");
const SOURCE_MANIFEST_FILE = path.join(FINGERPRINT_DIR, "fingerprints.manifest.json");
const SNAPSHOT_DIR = path.join(FINGERPRINT_DIR, "snapshots");
const SNAPSHOT_INDEX_FILE = path.join(SNAPSHOT_DIR, "snapshot-index.json");
const LATEST_RECORDS_FILE = path.join(SNAPSHOT_DIR, "latest.records.json");
const LATEST_MANIFEST_FILE = path.join(SNAPSHOT_DIR, "latest.manifest.json");

function assertExists(filePath, label) {
	if (!fs.existsSync(filePath)) {
		throw new Error(`${label} not found: ${filePath}`);
	}
}

function ensureDir(dirPath) {
	fs.mkdirSync(dirPath, { recursive: true });
}

function readJson(filePath, fallback = null) {
	if (!fs.existsSync(filePath)) {
		return fallback;
	}

	return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function copyFile(source, target) {
	fs.copyFileSync(source, target);
}

function timestampLabel() {
	const now = new Date();
	const yyyy = String(now.getFullYear());
	const mm = String(now.getMonth() + 1).padStart(2, "0");
	const dd = String(now.getDate()).padStart(2, "0");
	const hh = String(now.getHours()).padStart(2, "0");
	const mi = String(now.getMinutes()).padStart(2, "0");
	const ss = String(now.getSeconds()).padStart(2, "0");
	return `${yyyy}${mm}${dd}-${hh}${mi}${ss}`;
}

function parseArgs(argv) {
	const options = {
		label: "",
		keep: 20
	};

	for (let index = 0; index < argv.length; index += 1) {
		const arg = argv[index];
		const next = argv[index + 1];

		if (arg === "--label" && next) {
			options.label = next;
			index += 1;
			continue;
		}

		if (arg === "--keep" && next) {
			options.keep = Number(next);
			index += 1;
		}
	}

	if (!Number.isFinite(options.keep) || options.keep < 1) {
		throw new Error("Invalid --keep value. It must be a positive integer.");
	}

	return options;
}

function sanitizeLabel(label) {
	return String(label || "")
		.trim()
		.replace(/[^A-Za-z0-9._-]+/g, "_")
		.replace(/^_+|_+$/g, "");
}

function main() {
	const options = parseArgs(process.argv.slice(2));
	assertExists(SOURCE_RECORDS_FILE, "Fingerprint records");
	assertExists(SOURCE_MANIFEST_FILE, "Fingerprint manifest");
	ensureDir(SNAPSHOT_DIR);

	const baseLabel = sanitizeLabel(options.label) || timestampLabel();
	const recordsFileName = `fingerprints.${baseLabel}.records.json`;
	const manifestFileName = `fingerprints.${baseLabel}.manifest.json`;
	const snapshotRecordsPath = path.join(SNAPSHOT_DIR, recordsFileName);
	const snapshotManifestPath = path.join(SNAPSHOT_DIR, manifestFileName);

	copyFile(SOURCE_RECORDS_FILE, snapshotRecordsPath);
	copyFile(SOURCE_MANIFEST_FILE, snapshotManifestPath);
	copyFile(SOURCE_RECORDS_FILE, LATEST_RECORDS_FILE);
	copyFile(SOURCE_MANIFEST_FILE, LATEST_MANIFEST_FILE);

	const manifest = readJson(SOURCE_MANIFEST_FILE, {});
	const existingIndex = readJson(SNAPSHOT_INDEX_FILE, { snapshots: [] });
	const snapshots = Array.isArray(existingIndex.snapshots) ? existingIndex.snapshots : [];

	snapshots.push({
		label: baseLabel,
		created_at: new Date().toISOString(),
		records_file: path.relative(PUBLIC_DIR, snapshotRecordsPath),
		manifest_file: path.relative(PUBLIC_DIR, snapshotManifestPath),
		record_count: manifest.record_count ?? null,
		duplicate_content_group_count: manifest.duplicate_content_group_count ?? null
	});

	snapshots.sort((left, right) => right.created_at.localeCompare(left.created_at));

	const removed = [];
	while (snapshots.length > options.keep) {
		const stale = snapshots.pop();
		if (!stale) {
			break;
		}

		removed.push(stale);
	}

	for (const stale of removed) {
		const staleRecordsPath = path.join(PUBLIC_DIR, stale.records_file);
		const staleManifestPath = path.join(PUBLIC_DIR, stale.manifest_file);
		if (fs.existsSync(staleRecordsPath)) {
			fs.unlinkSync(staleRecordsPath);
		}
		if (fs.existsSync(staleManifestPath)) {
			fs.unlinkSync(staleManifestPath);
		}
	}

	const indexPayload = {
		latest_label: baseLabel,
		keep_limit: options.keep,
		latest_records_file: path.relative(PUBLIC_DIR, LATEST_RECORDS_FILE),
		latest_manifest_file: path.relative(PUBLIC_DIR, LATEST_MANIFEST_FILE),
		snapshot_count: snapshots.length,
		snapshots
	};

	fs.writeFileSync(SNAPSHOT_INDEX_FILE, `${JSON.stringify(indexPayload, null, 2)}\n`, "utf8");

	console.log(
		JSON.stringify(
			{
				snapshot_dir: path.relative(PUBLIC_DIR, SNAPSHOT_DIR),
				snapshot_index_file: path.relative(PUBLIC_DIR, SNAPSHOT_INDEX_FILE),
				latest_records_file: path.relative(PUBLIC_DIR, LATEST_RECORDS_FILE),
				latest_manifest_file: path.relative(PUBLIC_DIR, LATEST_MANIFEST_FILE),
				saved_label: baseLabel,
				snapshot_count: snapshots.length,
				removed_count: removed.length
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
