import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PUBLIC_DIR = path.resolve(__dirname, "..", "..", "..", "public");
const PARTS_DIR = path.join(PUBLIC_DIR, "bundle.parts");
const MANIFEST_FILE = path.join(PARTS_DIR, "manifest.json");
const OUTPUT_DIR = path.join(PUBLIC_DIR, "bundle.parts.index");
const OUTPUT_BUNDLE = path.join(OUTPUT_DIR, "rebuilt.bundle.js");
const OUTPUT_REPORT = path.join(OUTPUT_DIR, "rebuild-report.json");

function sha256(buffer) {
	return crypto.createHash("sha256").update(buffer).digest("hex");
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

function main() {
	assertExists(MANIFEST_FILE, "Manifest");
	ensureDir(OUTPUT_DIR);

	const manifest = readJson(MANIFEST_FILE);
	if (!Array.isArray(manifest.parts) || manifest.parts.length === 0) {
		throw new Error("manifest.json does not contain any parts.");
	}

	const buffers = [];
	const parts = [];

	for (const partName of manifest.parts) {
		const partPath = path.join(PARTS_DIR, partName);
		assertExists(partPath, "Bundle part");

		const buffer = fs.readFileSync(partPath);
		buffers.push(buffer);
		parts.push({
			file: partName,
			size_bytes: buffer.length,
			sha256: sha256(buffer)
		});
	}

	const rebuilt = Buffer.concat(buffers);
	const rebuiltSha = sha256(rebuilt);

	fs.writeFileSync(OUTPUT_BUNDLE, rebuilt);

	const report = {
		source: manifest.source ?? null,
		record_model: manifest.record_model ?? null,
		declared_part_count: manifest.part_count ?? null,
		actual_part_count: manifest.parts.length,
		target_part_size_bytes: manifest.target_part_size_bytes ?? null,
		max_part_size_bytes: manifest.max_part_size_bytes ?? null,
		largest_part_size_bytes: Math.max(...parts.map((part) => part.size_bytes)),
		rebuilt_file: path.relative(PUBLIC_DIR, OUTPUT_BUNDLE),
		rebuilt_size_bytes: rebuilt.length,
		rebuilt_sha256: rebuiltSha,
		part_count_match:
			manifest.part_count === undefined || manifest.part_count === manifest.parts.length,
		parts
	};

	fs.writeFileSync(OUTPUT_REPORT, `${JSON.stringify(report, null, 2)}\n`, "utf8");

	console.log(
		JSON.stringify(
			{
				rebuilt_file: path.relative(PUBLIC_DIR, OUTPUT_BUNDLE),
				report_file: path.relative(PUBLIC_DIR, OUTPUT_REPORT),
				part_count: manifest.parts.length,
				size_bytes: rebuilt.length,
				sha256: rebuiltSha
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
