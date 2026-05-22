import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as parser from "@babel/parser";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PUBLIC_DIR = path.resolve(__dirname, "..", "..", "..", "public");
const OUTPUT_DIR = path.join(PUBLIC_DIR, "bundle.parts.index");
const INPUT_FILE = path.join(OUTPUT_DIR, "rebuilt.bundle.js");
const STRUCTURE_INDEX_FILE = path.join(OUTPUT_DIR, "index.structure.raw.json");
const SYMBOL_INDEX_FILE = path.join(OUTPUT_DIR, "index.symbols.raw.json");
const REPORT_FILE = path.join(OUTPUT_DIR, "ast-scan-report.json");

function sha256(text) {
	return crypto.createHash("sha256").update(text).digest("hex");
}

function sizeOf(text) {
	return Buffer.byteLength(text, "utf8");
}

function assertExists(filePath, label) {
	if (!fs.existsSync(filePath)) {
		throw new Error(`${label} not found: ${filePath}`);
	}
}

function ensureDir(dirPath) {
	fs.mkdirSync(dirPath, { recursive: true });
}

function getPatternName(pattern) {
	if (!pattern) {
		return null;
	}

	if (pattern.type === "Identifier") {
		return pattern.name;
	}

	if (pattern.type === "RestElement") {
		return getPatternName(pattern.argument);
	}

	if (pattern.type === "AssignmentPattern") {
		return getPatternName(pattern.left);
	}

	if (pattern.type === "ObjectPattern") {
		return pattern.properties
			.map((property) => getPatternName(property.value || property.argument || property.key))
			.filter(Boolean)
			.join(", ");
	}

	if (pattern.type === "ArrayPattern") {
		return pattern.elements.map((item) => getPatternName(item)).filter(Boolean).join(", ");
	}

	return null;
}

function getDeclarationName(node) {
	if (!node) {
		return null;
	}

	if (node.id?.name) {
		return node.id.name;
	}

	if (node.declaration?.id?.name) {
		return node.declaration.id.name;
	}

	if (node.type === "VariableDeclaration") {
		return node.declarations.map((item) => getPatternName(item.id)).filter(Boolean).join(", ");
	}

	if (node.type === "ExportNamedDeclaration") {
		if (node.declaration) {
			return getDeclarationName(node.declaration);
		}

		return node.specifiers
			.map((specifier) => specifier.exported?.name || specifier.local?.name)
			.filter(Boolean)
			.join(", ");
	}

	if (node.type === "ExportDefaultDeclaration") {
		return getDeclarationName(node.declaration) || "default";
	}

	return null;
}

function extractSymbolsFromTopLevelNode(node) {
	const symbols = [];

	if (node.type === "FunctionDeclaration" && node.id?.name) {
		symbols.push({
			name: node.id.name,
			kind: "function",
			start: node.start,
			end: node.end
		});
	}

	if (node.type === "ClassDeclaration" && node.id?.name) {
		symbols.push({
			name: node.id.name,
			kind: "class",
			start: node.start,
			end: node.end
		});
	}

	if (node.type === "VariableDeclaration") {
		for (const declaration of node.declarations) {
			const name = getPatternName(declaration.id);
			if (name) {
				symbols.push({
					name,
					kind: node.kind,
					start: declaration.start,
					end: declaration.end
				});
			}
		}
	}

	if (
		node.type === "ExportNamedDeclaration" ||
		node.type === "ExportDefaultDeclaration"
	) {
		if (node.declaration) {
			symbols.push(...extractSymbolsFromTopLevelNode(node.declaration));
		}

		for (const specifier of node.specifiers || []) {
			const name = specifier.exported?.name || specifier.local?.name;
			if (name) {
				symbols.push({
					name,
					kind: "export",
					start: specifier.start,
					end: specifier.end
				});
			}
		}
	}

	return symbols;
}

function detectTags(source) {
	const tags = [];

	if (
		source.includes("user_effect") ||
		source.includes("template_effect") ||
		source.includes("append_styles")
	) {
		tags.push("svelte-compiled");
	}

	if (source.includes("hljs") || source.includes("highlight")) {
		tags.push("highlightjs");
	}

	if (
		source.includes("KaTeX") ||
		source.includes("mathmlBuilder") ||
		source.includes("htmlBuilder")
	) {
		tags.push("katex");
	}

	if (
		source.includes("pdfjs") ||
		source.includes("PDFDataRangeTransport") ||
		source.includes("WebAssembly")
	) {
		tags.push("pdfjs");
	}

	if (
		source.includes("ChatMessage") ||
		source.includes("sendMessage") ||
		source.includes("message.content")
	) {
		tags.push("chat-ui");
	}

	if (
		source.includes("Model Context Protocol") ||
		source.includes("mcp") ||
		source.includes("serverResources")
	) {
		tags.push("mcp");
	}

	if (source.includes("safeParse") || source.includes("Zod")) {
		tags.push("zod");
	}

	return [...new Set(tags)];
}

function getImportInfo(node) {
	if (node.type !== "ImportDeclaration") {
		return null;
	}

	return {
		source: node.source?.value ?? null,
		specifiers: node.specifiers.map((specifier) => ({
			kind: specifier.type,
			local: specifier.local?.name ?? null,
			imported: specifier.imported?.name ?? null
		}))
	};
}

function getExportInfo(node) {
	if (
		node.type !== "ExportNamedDeclaration" &&
		node.type !== "ExportDefaultDeclaration" &&
		node.type !== "ExportAllDeclaration"
	) {
		return null;
	}

	return {
		kind: node.type,
		source: node.source?.value ?? null,
		names: [
			...extractSymbolsFromTopLevelNode(node).map((item) => item.name),
			...(node.specifiers || [])
				.map((specifier) => specifier.exported?.name || specifier.local?.name)
				.filter(Boolean)
		].filter(Boolean)
	};
}

function classifyNode(node) {
	if (node.type === "VariableDeclaration") {
		return `${node.kind}_declaration`;
	}

	if (node.type === "FunctionDeclaration") {
		return "function_declaration";
	}

	if (node.type === "ClassDeclaration") {
		return "class_declaration";
	}

	if (node.type === "ImportDeclaration") {
		return "import_declaration";
	}

	if (node.type === "ExportNamedDeclaration") {
		return "export_named_declaration";
	}

	if (node.type === "ExportDefaultDeclaration") {
		return "export_default_declaration";
	}

	if (node.type === "ExpressionStatement") {
		return "expression_statement";
	}

	return node.type;
}

function main() {
	ensureDir(OUTPUT_DIR);
	assertExists(INPUT_FILE, "Rebuilt bundle");

	const code = fs.readFileSync(INPUT_FILE, "utf8");
	const ast = parser.parse(code, {
		sourceType: "unambiguous",
		errorRecovery: true,
		allowReturnOutsideFunction: true,
		plugins: [
			"jsx",
			"typescript",
			"classProperties",
			"classPrivateProperties",
			"classPrivateMethods",
			"dynamicImport",
			"importMeta",
			"topLevelAwait",
			"optionalChaining",
			"nullishCoalescingOperator",
			"objectRestSpread"
		]
	});

	const structureIndex = [];
	const symbolIndex = [];

	for (let index = 0; index < ast.program.body.length; index += 1) {
		const node = ast.program.body[index];
		const source = code.slice(node.start, node.end);
		const symbols = extractSymbolsFromTopLevelNode(node);

		const record = {
			id: `top_${String(index).padStart(6, "0")}`,
			order: index,
			kind: classifyNode(node),
			ast_type: node.type,
			name: getDeclarationName(node),
			start: node.start,
			end: node.end,
			size_bytes: sizeOf(source),
			line_start: node.loc?.start?.line ?? null,
			line_end: node.loc?.end?.line ?? null,
			tags: detectTags(source),
			symbols: symbols.map((item) => item.name),
			import: getImportInfo(node),
			export: getExportInfo(node),
			sha256: sha256(source)
		};

		structureIndex.push(record);

		for (const symbol of symbols) {
			symbolIndex.push({
				symbol: symbol.name,
				kind: symbol.kind,
				top_level_id: record.id,
				top_level_kind: record.kind,
				start: symbol.start,
				end: symbol.end,
				parent_start: record.start,
				parent_end: record.end,
				parent_size_bytes: record.size_bytes,
				tags: record.tags
			});
		}
	}

	const report = {
		input_file: path.relative(PUBLIC_DIR, INPUT_FILE),
		input_size_bytes: sizeOf(code),
		top_level_node_count: structureIndex.length,
		symbol_count: symbolIndex.length,
		largest_nodes: [...structureIndex]
			.sort((left, right) => right.size_bytes - left.size_bytes)
			.slice(0, 30)
			.map((item) => ({
				id: item.id,
				kind: item.kind,
				name: item.name,
				size_bytes: item.size_bytes,
				tags: item.tags
			})),
		tag_counts: structureIndex.reduce((accumulator, item) => {
			for (const tag of item.tags) {
				accumulator[tag] = (accumulator[tag] || 0) + 1;
			}

			return accumulator;
		}, {})
	};

	fs.writeFileSync(STRUCTURE_INDEX_FILE, `${JSON.stringify(structureIndex, null, 2)}\n`, "utf8");
	fs.writeFileSync(SYMBOL_INDEX_FILE, `${JSON.stringify(symbolIndex, null, 2)}\n`, "utf8");
	fs.writeFileSync(REPORT_FILE, `${JSON.stringify(report, null, 2)}\n`, "utf8");

	console.log(
		JSON.stringify(
			{
				structure_file: path.relative(PUBLIC_DIR, STRUCTURE_INDEX_FILE),
				symbol_file: path.relative(PUBLIC_DIR, SYMBOL_INDEX_FILE),
				report_file: path.relative(PUBLIC_DIR, REPORT_FILE),
				top_level_node_count: structureIndex.length,
				symbol_count: symbolIndex.length
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
