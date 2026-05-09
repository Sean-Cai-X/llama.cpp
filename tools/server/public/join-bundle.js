#!/usr/bin/env node

const fs = require("fs");
const path = require("path");

const PUBLIC_DIR = __dirname;
const PARTS_DIR = path.join(PUBLIC_DIR, "bundle.parts");
const MANIFEST_FILE = path.join(PARTS_DIR, "manifest.json");
const OUTPUT_FILE = path.join(PUBLIC_DIR, "bundle.joined.js");

function readManifest() {
	if (!fs.existsSync(MANIFEST_FILE)) {
		throw new Error(`Manifest not found: ${MANIFEST_FILE}`);
	}

	const manifest = JSON.parse(fs.readFileSync(MANIFEST_FILE, "utf8"));
	if (!manifest || !Array.isArray(manifest.parts) || manifest.parts.length === 0) {
		throw new Error("Manifest does not contain any bundle parts.");
	}

	return manifest;
}

function main() {
	const manifest = readManifest();
	const joined = manifest.parts
		.map((fileName) => {
			const filePath = path.join(PARTS_DIR, fileName);
			if (!fs.existsSync(filePath)) {
				throw new Error(`Missing bundle part listed in manifest: ${fileName}`);
			}
			return fs.readFileSync(filePath, "utf8");
		})
		.join("");

	fs.writeFileSync(OUTPUT_FILE, joined, "utf8");

	console.log(
		JSON.stringify(
			{
				output: path.basename(OUTPUT_FILE),
				part_count: manifest.parts.length,
				size_bytes: Buffer.byteLength(joined, "utf8")
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
