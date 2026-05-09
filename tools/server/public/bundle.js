const manifestUrl = new URL("./bundle.parts/manifest.json", import.meta.url);

async function loadManifest() {
	const response = await fetch(manifestUrl);
	if (!response.ok) {
		throw new Error(`Failed to load bundle manifest: ${response.status} ${response.statusText}`);
	}

	return response.json();
}

async function loadBundleModule() {
	const manifest = await loadManifest();
	if (!manifest || !Array.isArray(manifest.parts) || manifest.parts.length === 0) {
		throw new Error("Bundle manifest is missing part entries.");
	}

	const sourceTexts = await Promise.all(
		manifest.parts.map(async (partName) => {
			const partUrl = new URL(`./bundle.parts/${partName}`, import.meta.url);
			const response = await fetch(partUrl);
			if (!response.ok) {
				throw new Error(`Failed to load bundle part ${partName}: ${response.status} ${response.statusText}`);
			}
			return response.text();
		})
	);

	const sourceText = sourceTexts.join("");
	const blob = new Blob([sourceText, "\n//# sourceURL=bundle.reconstructed.js\n"], {
		type: "text/javascript"
	});
	const blobUrl = URL.createObjectURL(blob);

	try {
		return await import(blobUrl);
	} finally {
		URL.revokeObjectURL(blobUrl);
	}
}

let bundleModule;

try {
	bundleModule = await loadBundleModule();
} catch (error) {
	console.error("[bundle-loader] Failed to rebuild bundle.js from bundle.parts.", error);
	throw error;
}

const { app, start } = bundleModule;

export { app, start };
