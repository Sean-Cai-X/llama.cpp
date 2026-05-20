import type { ChatMessageTimings, ChatRole, ChatMessageType } from '$lib/types/chat';
import { AttachmentType } from '$lib/enums';

export interface McpServerOverride {
	serverId: string;
	enabled: boolean;
}

export interface StructuredConclusion {
	task_state?: string;
	reasoning_level?: string;
	primary_intent?: string;
	secondary_intents?: string[];
	intent_confidence?: number;
	association_scope?: string;
	entity_refs?: string[];
	evidence_refs?: string[];
	risk_flags?: string[];
	next_action?: string;
	session_id?: string;
	turn_id?: string;
	slice_summary?: string;
	expression_keys?: string[];
	query?: string;
	arguments_text?: string;
	summary?: string;
	insufficient_context?: boolean;
	producer?: string;
	stage?: string;
}

export interface DialogSliceRecord {
	sessionId: string;
	turnId: string;
	sliceId?: string;
	sliceVersion?: string;
	auditRef?: string;
	userText: string;
	assistantText: string;
	summary?: string;
	dedupStatus?: string;
	canonicalSliceId?: string;
	sliceRefs?: string[];
	storageRefs?: string[];
	expressionKeys?: string[];
	evidenceRefs?: string[];
	createdAt: number;
}

export interface ExecutionBindingRecord {
	sessionId: string;
	turnId: string;
	primaryIntent?: string;
	taskId?: string;
	taskIds?: string[];
	taskGroupId?: string;
	sourceType?: string;
	sourceLabel?: string;
	sourceDetail?: string;
	handoffFrom?: string;
	handoffTo?: string;
	takeoverRelation?: string;
	evidenceRefs?: string[];
	logPath?: string;
	toolName?: string;
	status?: string;
}

export interface DatabaseConversation {
	currNode: string | null;
	id: string;
	lastModified: number;
	name: string;
	sessionId?: string;
	lastTurnId?: string;
	currentTaskState?: string;
	currentReasoningLevel?: string;
	currentPrimaryIntent?: string;
	currentIntentConfidence?: number;
	currentSummary?: string;
	sessionSemanticProjectionReady?: boolean;
	sessionSemanticProjectionSource?: string;
	semanticBindingMode?: string;
	semanticObservabilityMode?: string;
	semanticCatalogCount?: number;
	mcp18080Count?: number;
	semantic8095Count?: number;
	remoteDialogSemanticListCount?: number;
	callableSemanticCount?: number;
	nonCallableSemanticCount?: number;
	mountedToolCount?: number;
	displayProjectionMode?: string;
	allCatalogEntriesVisibleInDialogList?: boolean;
	catalogIsSingleSourceOfTruth?: boolean;
	availableToolClasses?: string[];
	lastTaskId?: string;
	currentTaskGroupId?: string;
	lastEvidenceRefs?: string[];
	currentExpressionKeys?: string[];
	currentSourceType?: string;
	currentSourceLabel?: string;
	currentSourceDetail?: string;
	currentHandoffFrom?: string;
	currentHandoffTo?: string;
	currentTakeoverRelation?: string;
	mcpServerOverrides?: McpServerOverride[];
	forkedFromConversationId?: string;
}

export interface DatabaseMessageExtraAudioFile {
	type: AttachmentType.AUDIO;
	name: string;
	base64Data: string;
	mimeType: string;
}

export interface DatabaseMessageExtraImageFile {
	type: AttachmentType.IMAGE;
	name: string;
	base64Url: string;
}

/**
 * Legacy format from old webui - pasted content was stored as "context" type
 * @deprecated Use DatabaseMessageExtraTextFile instead
 */
export interface DatabaseMessageExtraLegacyContext {
	type: AttachmentType.LEGACY_CONTEXT;
	name: string;
	content: string;
}

export interface DatabaseMessageExtraPdfFile {
	type: AttachmentType.PDF;
	base64Data: string;
	name: string;
	content: string;
	images?: string[];
	processedAsImages: boolean;
}

export interface DatabaseMessageExtraTextFile {
	type: AttachmentType.TEXT;
	name: string;
	content: string;
}

export interface DatabaseMessageExtraMcpPrompt {
	type: AttachmentType.MCP_PROMPT;
	name: string;
	serverName: string;
	promptName: string;
	content: string;
	arguments?: Record<string, string>;
}

export interface DatabaseMessageExtraMcpResource {
	type: AttachmentType.MCP_RESOURCE;
	name: string;
	uri: string;
	serverName: string;
	content: string;
	mimeType?: string;
}

export type DatabaseMessageExtra =
	| DatabaseMessageExtraImageFile
	| DatabaseMessageExtraTextFile
	| DatabaseMessageExtraAudioFile
	| DatabaseMessageExtraPdfFile
	| DatabaseMessageExtraMcpPrompt
	| DatabaseMessageExtraMcpResource
	| DatabaseMessageExtraLegacyContext;

export interface DatabaseMessage {
	id: string;
	convId: string;
	sessionId?: string;
	turnId?: string;
	taskId?: string;
	taskGroupId?: string;
	intentKey?: string;
	evidenceRefs?: string[];
	sourceType?: string;
	sourceLabel?: string;
	sourceDetail?: string;
	handoffFrom?: string;
	handoffTo?: string;
	takeoverRelation?: string;
	type: ChatMessageType;
	timestamp: number;
	role: ChatRole;
	content: string;
	parent: string | null;
	/**
	 * @deprecated - left for backward compatibility
	 */
	thinking?: string;
	/** Reasoning content produced by the model (separate from visible content) */
	reasoningContent?: string;
	/** Serialized JSON array of tool calls made by assistant messages */
	toolCalls?: string;
	/** Tool call ID for tool result messages (role: 'tool') */
	toolCallId?: string;
	children: string[];
	extra?: DatabaseMessageExtra[];
	timings?: ChatMessageTimings;
	model?: string;
	structuredConclusion?: StructuredConclusion;
	dialogSlice?: DialogSliceRecord;
	executionBinding?: ExecutionBindingRecord;
}

export type ExportedConversation = {
	conv: DatabaseConversation;
	messages: DatabaseMessage[];
};

export type ExportedConversations = ExportedConversation | ExportedConversation[];
