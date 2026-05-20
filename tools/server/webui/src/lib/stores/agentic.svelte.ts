/**
 * agenticStore - Reactive State Store for Agentic Loop Orchestration
 *
 * Manages multi-turn agentic loop with MCP tools:
 * - LLM streaming with tool call detection
 * - Tool execution via mcpStore
 * - Session state management
 * - Turn limit enforcement
 *
 * Each agentic turn produces separate DB messages:
 * - One assistant message per LLM turn (with tool_calls if any)
 * - One tool result message per tool call execution
 *
 * **Architecture & Relationships:**
 * - **ChatService**: Stateless API layer (sendMessage, streaming)
 * - **mcpStore**: MCP connection management and tool execution
 * - **agenticStore** (this): Reactive state + business logic
 *
 * @see ChatService in services/chat.service.ts for API operations
 * @see mcpStore in stores/mcp.svelte.ts for MCP operations
 */

import { ChatService } from '$lib/services';
import { config } from '$lib/stores/settings.svelte';
import { mcpStore } from '$lib/stores/mcp.svelte';
import { modelsStore } from '$lib/stores/models.svelte';
import { isAbortError } from '$lib/utils';
import {
	DEFAULT_AGENTIC_CONFIG,
	NEWLINE_SEPARATOR,
	TURN_LIMIT_MESSAGE,
	LLM_ERROR_BLOCK_START,
	LLM_ERROR_BLOCK_END
} from '$lib/constants';
import {
	IMAGE_MIME_TO_EXTENSION,
	DATA_URI_BASE64_REGEX,
	MCP_ATTACHMENT_NAME_PREFIX,
	DEFAULT_IMAGE_EXTENSION
} from '$lib/constants';
import {
	AttachmentType,
	ContentPartType,
	MessageRole,
	MimeTypePrefix,
	ToolCallType
} from '$lib/enums';
import type {
	AgenticFlowParams,
	AgenticFlowResult,
	AgenticSession,
	AgenticConfig,
	SettingsConfigType,
	McpServerOverride,
	MCPToolCall
} from '$lib/types';
import type {
	AgenticMessage,
	AgenticToolCallList,
	AgenticFlowCallbacks,
	AgenticFlowOptions
} from '$lib/types/agentic';
import type {
	ApiChatCompletionToolCall,
	ApiChatMessageData,
	ApiChatMessageContentPart
} from '$lib/types/api';
import type {
	ChatMessagePromptProgress,
	ChatMessageTimings,
	ChatMessageAgenticTimings,
	ChatMessageToolCallTiming,
	ChatMessageAgenticTurnStats
} from '$lib/types/chat';
import type {
	DatabaseMessage,
	DatabaseMessageExtra,
	DatabaseMessageExtraImageFile
} from '$lib/types/database';

function createDefaultSession(): AgenticSession {
	return {
		isRunning: false,
		currentTurn: 0,
		totalToolCalls: 0,
		lastError: null,
		streamingToolCall: null
	};
}

function toAgenticMessages(messages: ApiChatMessageData[]): AgenticMessage[] {
	return messages.map((message) => {
		if (
			message.role === MessageRole.ASSISTANT &&
			message.tool_calls &&
			message.tool_calls.length > 0
		) {
			return {
				role: MessageRole.ASSISTANT,
				content: message.content,
				tool_calls: message.tool_calls.map((call, index) => ({
					id: call.id ?? `call_${index}`,
					type: (call.type as ToolCallType.FUNCTION) ?? ToolCallType.FUNCTION,
					function: { name: call.function?.name ?? '', arguments: call.function?.arguments ?? '' }
				}))
			} satisfies AgenticMessage;
		}
		if (message.role === MessageRole.TOOL && message.tool_call_id) {
			return {
				role: MessageRole.TOOL,
				tool_call_id: message.tool_call_id,
				content: typeof message.content === 'string' ? message.content : ''
			} satisfies AgenticMessage;
		}
		return {
			role: message.role as MessageRole.SYSTEM | MessageRole.USER,
			content: message.content
		} satisfies AgenticMessage;
	});
}

type ContinuationPayload = {
	status?: string;
	acceptance_status?: string;
	task_completion?: string;
	batch_completion?: string;
	content_read_completion?: string;
	goal_status?: string;
	continue_required?: boolean;
	auto_continue_required?: boolean;
	assistant_response_allowed?: boolean;
	final_answer_allowed?: boolean;
	next_call_json?: unknown;
	required_tool_arguments_json?: unknown;
	next_action_0_tool_name?: string;
	next_action_0_params_json?: string;
};

type ContinuationExecutionResult = {
	content: string;
	attachments: DatabaseMessageExtra[];
	executedCalls: number;
};

class AgenticStore {
	private _sessions = $state<Map<string, AgenticSession>>(new Map());
	private static readonly MAX_CONTINUATION_STEPS = 1024;

	get isReady(): boolean {
		return true;
	}
	get isAnyRunning(): boolean {
		for (const session of this._sessions.values()) {
			if (session.isRunning) return true;
		}
		return false;
	}

	getSession(conversationId: string): AgenticSession {
		let session = this._sessions.get(conversationId);
		if (!session) {
			session = createDefaultSession();
			this._sessions.set(conversationId, session);
		}
		return session;
	}

	private updateSession(conversationId: string, update: Partial<AgenticSession>): void {
		const session = this.getSession(conversationId);
		this._sessions.set(conversationId, { ...session, ...update });
	}

	clearSession(conversationId: string): void {
		this._sessions.delete(conversationId);
	}

	getActiveSessions(): Array<{ conversationId: string; session: AgenticSession }> {
		const active: Array<{ conversationId: string; session: AgenticSession }> = [];
		for (const [conversationId, session] of this._sessions.entries()) {
			if (session.isRunning) active.push({ conversationId, session });
		}
		return active;
	}

	isRunning(conversationId: string): boolean {
		return this.getSession(conversationId).isRunning;
	}

	currentTurn(conversationId: string): number {
		return this.getSession(conversationId).currentTurn;
	}

	totalToolCalls(conversationId: string): number {
		return this.getSession(conversationId).totalToolCalls;
	}

	lastError(conversationId: string): Error | null {
		return this.getSession(conversationId).lastError;
	}

	streamingToolCall(conversationId: string): { name: string; arguments: string } | null {
		return this.getSession(conversationId).streamingToolCall;
	}

	clearError(conversationId: string): void {
		this.updateSession(conversationId, { lastError: null });
	}

	getConfig(settings: SettingsConfigType, perChatOverrides?: McpServerOverride[]): AgenticConfig {
		const maxTurns = Number(settings.agenticMaxTurns) || DEFAULT_AGENTIC_CONFIG.maxTurns;
		const maxToolPreviewLines =
			Number(settings.agenticMaxToolPreviewLines) || DEFAULT_AGENTIC_CONFIG.maxToolPreviewLines;
		return {
			enabled: mcpStore.hasEnabledServers(perChatOverrides) && DEFAULT_AGENTIC_CONFIG.enabled,
			maxTurns,
			maxToolPreviewLines
		};
	}

	async runAgenticFlow(params: AgenticFlowParams): Promise<AgenticFlowResult> {
		const { conversationId, messages, options = {}, callbacks, signal, perChatOverrides } = params;

		const agenticConfig = this.getConfig(config(), perChatOverrides);
		if (!agenticConfig.enabled) return { handled: false };

		const initialized = await mcpStore.ensureInitialized(perChatOverrides);
		if (!initialized) {
			console.log('[AgenticStore] MCP not initialized, falling back to standard chat');
			return { handled: false };
		}

		const tools = mcpStore.getToolDefinitionsForLLM();
		if (tools.length === 0) {
			console.log('[AgenticStore] No tools available, falling back to standard chat');
			return { handled: false };
		}

		console.log(`[AgenticStore] Starting agentic flow with ${tools.length} tools`);

		const normalizedMessages: ApiChatMessageData[] = messages
			.map((msg) => {
				if ('id' in msg && 'convId' in msg && 'timestamp' in msg)
					return ChatService.convertDbMessageToApiChatMessageData(
						msg as DatabaseMessage & { extra?: DatabaseMessageExtra[] }
					);
				return msg as ApiChatMessageData;
			})
			.filter((msg) => {
				if (msg.role === MessageRole.SYSTEM) {
					const content = typeof msg.content === 'string' ? msg.content : '';
					return content.trim().length > 0;
				}
				return true;
			});

		this.updateSession(conversationId, {
			isRunning: true,
			currentTurn: 0,
			totalToolCalls: 0,
			lastError: null
		});
		mcpStore.acquireConnection();

		try {
			await this.executeAgenticLoop({
				conversationId,
				messages: normalizedMessages,
				options,
				tools,
				agenticConfig,
				callbacks,
				signal
			});
			return { handled: true };
		} catch (error) {
			const normalizedError = error instanceof Error ? error : new Error(String(error));
			this.updateSession(conversationId, { lastError: normalizedError });
			callbacks.onError?.(normalizedError);
			return { handled: true, error: normalizedError };
		} finally {
			this.updateSession(conversationId, { isRunning: false });
			await mcpStore
				.releaseConnection()
				.catch((err: unknown) =>
					console.warn('[AgenticStore] Failed to release MCP connection:', err)
				);
		}
	}

	private async executeAgenticLoop(params: {
		conversationId: string;
		messages: ApiChatMessageData[];
		options: AgenticFlowOptions;
		tools: ReturnType<typeof mcpStore.getToolDefinitionsForLLM>;
		agenticConfig: AgenticConfig;
		callbacks: AgenticFlowCallbacks;
		signal?: AbortSignal;
	}): Promise<void> {
		const { conversationId, messages, options, tools, agenticConfig, callbacks, signal } = params;
		const {
			onChunk,
			onReasoningChunk,
			onToolCallsStreaming,
			onAttachments,
			onModel,
			onAssistantTurnComplete,
			createToolResultMessage,
			createAssistantMessage,
			onFlowComplete,
			onTimings,
			onTurnComplete
		} = callbacks;

		const sessionMessages: AgenticMessage[] = toAgenticMessages(messages);
		let capturedTimings: ChatMessageTimings | undefined;
		let totalToolCallCount = 0;

		const agenticTimings: ChatMessageAgenticTimings = {
			turns: 0,
			toolCallsCount: 0,
			toolsMs: 0,
			toolCalls: [],
			perTurn: [],
			llm: { predicted_n: 0, predicted_ms: 0, prompt_n: 0, prompt_ms: 0 }
		};
		const maxTurns = agenticConfig.maxTurns;

		const effectiveModel = options.model || modelsStore.models[0]?.model || '';

		for (let turn = 0; turn < maxTurns; turn++) {
			this.updateSession(conversationId, { currentTurn: turn + 1 });
			agenticTimings.turns = turn + 1;

			if (signal?.aborted) {
				onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));
				return;
			}

			// For turns > 0, create a new assistant message via callback
			if (turn > 0 && createAssistantMessage) {
				await createAssistantMessage();
			}

			let turnContent = '';
			let turnReasoningContent = '';
			let turnToolCalls: ApiChatCompletionToolCall[] = [];
			let lastStreamingToolCallName = '';
			let lastStreamingToolCallArgsLength = 0;
			let turnTimings: ChatMessageTimings | undefined;

			const turnStats: ChatMessageAgenticTurnStats = {
				turn: turn + 1,
				llm: { predicted_n: 0, predicted_ms: 0, prompt_n: 0, prompt_ms: 0 },
				toolCalls: [],
				toolsMs: 0
			};

			try {
				await ChatService.sendMessage(
					sessionMessages as ApiChatMessageData[],
					{
						...options,
						stream: true,
						tools: tools.length > 0 ? tools : undefined,
						onChunk: (chunk: string) => {
							turnContent += chunk;
							onChunk?.(chunk);
						},
						onReasoningChunk: (chunk: string) => {
							turnReasoningContent += chunk;
							onReasoningChunk?.(chunk);
						},
						onToolCallChunk: (serialized: string) => {
							try {
								turnToolCalls = JSON.parse(serialized) as ApiChatCompletionToolCall[];
								onToolCallsStreaming?.(turnToolCalls);

								if (turnToolCalls.length > 0 && turnToolCalls[0]?.function) {
									const name = turnToolCalls[0].function.name || '';
									const args = turnToolCalls[0].function.arguments || '';
									const argsLengthBucket = Math.floor(args.length / 100);
									if (
										name !== lastStreamingToolCallName ||
										argsLengthBucket !== lastStreamingToolCallArgsLength
									) {
										lastStreamingToolCallName = name;
										lastStreamingToolCallArgsLength = argsLengthBucket;
										this.updateSession(conversationId, {
											streamingToolCall: { name, arguments: args }
										});
									}
								}
							} catch {
								/* Ignore parse errors during streaming */
							}
						},
						onModel,
						onTimings: (timings?: ChatMessageTimings, progress?: ChatMessagePromptProgress) => {
							onTimings?.(timings, progress);
							if (timings) {
								capturedTimings = timings;
								turnTimings = timings;
							}
						},
						onComplete: () => {
							/* Completion handled after sendMessage resolves */
						},
						onError: (error: Error) => {
							throw error;
						}
					},
					undefined,
					signal
				);

				this.updateSession(conversationId, { streamingToolCall: null });

				if (turnTimings) {
					agenticTimings.llm.predicted_n += turnTimings.predicted_n || 0;
					agenticTimings.llm.predicted_ms += turnTimings.predicted_ms || 0;
					agenticTimings.llm.prompt_n += turnTimings.prompt_n || 0;
					agenticTimings.llm.prompt_ms += turnTimings.prompt_ms || 0;
					turnStats.llm.predicted_n = turnTimings.predicted_n || 0;
					turnStats.llm.predicted_ms = turnTimings.predicted_ms || 0;
					turnStats.llm.prompt_n = turnTimings.prompt_n || 0;
					turnStats.llm.prompt_ms = turnTimings.prompt_ms || 0;
				}
			} catch (error) {
				if (signal?.aborted) {
					// Save whatever we have for this turn before exiting
					await onAssistantTurnComplete?.(
						turnContent,
						turnReasoningContent || undefined,
						this.buildFinalTimings(capturedTimings, agenticTimings),
						undefined
					);
					onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));
					return;
				}
				const normalizedError = error instanceof Error ? error : new Error('LLM stream error');
				// Save error as content in the current turn
				onChunk?.(`${LLM_ERROR_BLOCK_START}${normalizedError.message}${LLM_ERROR_BLOCK_END}`);
				await onAssistantTurnComplete?.(
					turnContent + `${LLM_ERROR_BLOCK_START}${normalizedError.message}${LLM_ERROR_BLOCK_END}`,
					turnReasoningContent || undefined,
					this.buildFinalTimings(capturedTimings, agenticTimings),
					undefined
				);
				onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));
				throw normalizedError;
			}

			// No tool calls = final turn, save and complete
			if (turnToolCalls.length === 0) {
				agenticTimings.perTurn!.push(turnStats);

				const finalTimings = this.buildFinalTimings(capturedTimings, agenticTimings);

				await onAssistantTurnComplete?.(
					turnContent,
					turnReasoningContent || undefined,
					finalTimings,
					undefined
				);

				if (finalTimings) onTurnComplete?.(finalTimings);

				onFlowComplete?.(finalTimings);

				return;
			}

			// Normalize and save assistant turn with tool calls
			const normalizedCalls = this.normalizeToolCalls(turnToolCalls);
			if (normalizedCalls.length === 0) {
				await onAssistantTurnComplete?.(
					turnContent,
					turnReasoningContent || undefined,
					this.buildFinalTimings(capturedTimings, agenticTimings),
					undefined
				);
				onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));
				return;
			}
			console.info(
				`[AgenticStore] agentic_turn: tool_calls turn=${turn + 1} count=${normalizedCalls.length} names=${normalizedCalls.map((call) => call.function.name).join(',')}`
			);

			totalToolCallCount += normalizedCalls.length;
			this.updateSession(conversationId, { totalToolCalls: totalToolCallCount });

			// Save the assistant message with its tool calls
			await onAssistantTurnComplete?.(
				turnContent,
				turnReasoningContent || undefined,
				turnTimings,
				normalizedCalls
			);

			// Add assistant message to session history
			sessionMessages.push({
				role: MessageRole.ASSISTANT,
				content: turnContent || undefined,
				reasoning_content: turnReasoningContent || undefined,
				tool_calls: normalizedCalls
			});

			// Execute each tool call and create result messages
			for (const toolCall of normalizedCalls) {
				if (signal?.aborted) {
					onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));
					return;
				}
				console.info(
					`[AgenticStore] agentic_turn: dispatch tool_call_id=${toolCall.id} tool=${toolCall.function.name}`
				);
				const execution = await this.executeToolWithContinuation(
					conversationId,
					toolCall,
					agenticTimings,
					turnStats,
					totalToolCallCount,
					signal
				);
				totalToolCallCount += Math.max(0, execution.executedCalls - 1);
				this.updateSession(conversationId, { totalToolCalls: totalToolCallCount });
				console.info(
					`[AgenticStore] agentic_turn: completed tool_call_id=${toolCall.id} tool=${toolCall.function.name} executed_calls=${execution.executedCalls} result_bytes=${execution.content.length}`
				);

				if (signal?.aborted) {
					onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));
					return;
				}

				// Create the tool result message in the DB
				let toolResultMessage: DatabaseMessage | undefined;
				if (createToolResultMessage) {
					toolResultMessage = await createToolResultMessage(
						toolCall.id,
						execution.content,
						execution.attachments.length > 0 ? execution.attachments : undefined
					);
				}

				if (execution.attachments.length > 0 && toolResultMessage) {
					onAttachments?.(toolResultMessage.id, execution.attachments);
				}

				// Build content parts for session history (including images for vision models)
				const contentParts: ApiChatMessageContentPart[] = [
					{ type: ContentPartType.TEXT, text: execution.content }
				];
				for (const attachment of execution.attachments) {
					if (attachment.type === AttachmentType.IMAGE) {
						if (modelsStore.modelSupportsVision(effectiveModel)) {
							contentParts.push({
								type: ContentPartType.IMAGE_URL,
								image_url: { url: (attachment as DatabaseMessageExtraImageFile).base64Url }
							});
						} else {
							console.info(
								`[AgenticStore] Skipping image attachment (model "${effectiveModel}" does not support vision)`
							);
						}
					}
				}

				sessionMessages.push({
					role: MessageRole.TOOL,
					tool_call_id: toolCall.id,
					content: contentParts.length === 1 ? execution.content : contentParts
				});
			}

			if (turnStats.toolCalls.length > 0) {
				agenticTimings.perTurn!.push(turnStats);

				const intermediateTimings = this.buildFinalTimings(capturedTimings, agenticTimings);
				if (intermediateTimings) onTurnComplete?.(intermediateTimings);
			}

		}

		// Turn limit reached
		onChunk?.(TURN_LIMIT_MESSAGE);
		await onAssistantTurnComplete?.(
			TURN_LIMIT_MESSAGE,
			undefined,
			this.buildFinalTimings(capturedTimings, agenticTimings),
			undefined
		);
		onFlowComplete?.(this.buildFinalTimings(capturedTimings, agenticTimings));
	}

	private async executeToolWithContinuation(
		conversationId: string,
		toolCall: AgenticToolCallList[number],
		agenticTimings: ChatMessageAgenticTimings,
		turnStats: ChatMessageAgenticTurnStats,
		totalToolCallCount: number,
		signal?: AbortSignal
	): Promise<ContinuationExecutionResult> {
		let pendingCall: MCPToolCall | null = {
			id: toolCall.id,
			function: {
				name: toolCall.function.name,
				arguments: toolCall.function.arguments
			}
		};
		let continuationIndex = 0;
		let executedCalls = 0;
		const resultChunks: string[] = [];
		const attachments: DatabaseMessageExtra[] = [];

		while (pendingCall && continuationIndex < AgenticStore.MAX_CONTINUATION_STEPS) {
			const toolStartTime = performance.now();
			let result = '';
			let toolSuccess = true;

			try {
				console.info(
					`[AgenticStore] agentic_tool_execute: start step=${continuationIndex} tool=${pendingCall.function.name}`
				);
				const executionResult = await mcpStore.executeTool(pendingCall, signal);
				result = executionResult.content;
				toolSuccess = !executionResult.isError;
				console.info(
					`[AgenticStore] agentic_tool_execute: result step=${continuationIndex} tool=${pendingCall.function.name} success=${toolSuccess} bytes=${result.length}`
				);
			} catch (error) {
				if (isAbortError(error)) {
					throw error;
				}
				result = `Error: ${error instanceof Error ? error.message : String(error)}`;
				toolSuccess = false;
				console.warn(
					`[AgenticStore] agentic_tool_execute: error step=${continuationIndex} tool=${pendingCall.function.name}`,
					error
				);
			}

			const toolDurationMs = performance.now() - toolStartTime;
			const toolTiming: ChatMessageToolCallTiming = {
				name: pendingCall.function.name,
				duration_ms: Math.round(toolDurationMs),
				success: toolSuccess
			};

			agenticTimings.toolCalls!.push(toolTiming);
			agenticTimings.toolCallsCount++;
			agenticTimings.toolsMs += Math.round(toolDurationMs);
			turnStats.toolCalls.push(toolTiming);
			turnStats.toolsMs += Math.round(toolDurationMs);

			executedCalls++;
			if (executedCalls > 1) {
				totalToolCallCount++;
				this.updateSession(conversationId, { totalToolCalls: totalToolCallCount });
			}

			const extracted = this.extractBase64Attachments(result);
			if (extracted.cleanedResult.trim()) {
				resultChunks.push(extracted.cleanedResult);
			}
			attachments.push(...extracted.attachments);

			if (!toolSuccess) {
				pendingCall = null;
				break;
			}

			pendingCall = this.extractContinuationToolCall(
				extracted.cleanedResult,
				`${toolCall.id}_cont_${continuationIndex + 1}`
			);
			if (pendingCall) {
				console.info(
					`[AgenticStore] agentic_tool_execute: continuation next_step=${continuationIndex + 1} tool=${pendingCall.function.name}`
				);
			}
			continuationIndex++;
		}

		if (continuationIndex >= AgenticStore.MAX_CONTINUATION_STEPS) {
			resultChunks.push('Error: MCP continuation exceeded max steps');
		}

		return {
			content: resultChunks.join('\n\n'),
			attachments,
			executedCalls
		};
	}

	private buildFinalTimings(
		capturedTimings: ChatMessageTimings | undefined,
		agenticTimings: ChatMessageAgenticTimings
	): ChatMessageTimings | undefined {
		if (agenticTimings.toolCallsCount === 0) return capturedTimings;
		return {
			predicted_n: capturedTimings?.predicted_n,
			predicted_ms: capturedTimings?.predicted_ms,
			prompt_n: capturedTimings?.prompt_n,
			prompt_ms: capturedTimings?.prompt_ms,
			cache_n: capturedTimings?.cache_n,
			agentic: agenticTimings
		};
	}

	private normalizeToolCalls(toolCalls: ApiChatCompletionToolCall[]): AgenticToolCallList {
		if (!toolCalls) return [];
		return toolCalls.map((call, index) => ({
			id: call?.id ?? `tool_${index}`,
			type: (call?.type as ToolCallType.FUNCTION) ?? ToolCallType.FUNCTION,
			function: { name: call?.function?.name ?? '', arguments: call?.function?.arguments ?? '' }
		}));
	}

	private extractBase64Attachments(result: string): {
		cleanedResult: string;
		attachments: DatabaseMessageExtra[];
	} {
		if (!result.trim()) {
			return { cleanedResult: result, attachments: [] };
		}

		const lines = result.split(NEWLINE_SEPARATOR);
		const attachments: DatabaseMessageExtra[] = [];
		let attachmentIndex = 0;

		const cleanedLines = lines.map((line) => {
			const trimmedLine = line.trim();

			const match = trimmedLine.match(DATA_URI_BASE64_REGEX);
			if (!match) {
				return line;
			}

			const mimeType = match[1].toLowerCase();
			const base64Data = match[2];

			if (!base64Data) {
				return line;
			}

			attachmentIndex += 1;
			const name = this.buildAttachmentName(mimeType, attachmentIndex);

			if (mimeType.startsWith(MimeTypePrefix.IMAGE)) {
				attachments.push({ type: AttachmentType.IMAGE, name, base64Url: trimmedLine });

				return `[Attachment saved: ${name}]`;
			}

			return line;
		});

		return { cleanedResult: cleanedLines.join(NEWLINE_SEPARATOR), attachments };
	}

	private buildAttachmentName(mimeType: string, index: number): string {
		const extension = IMAGE_MIME_TO_EXTENSION[mimeType] ?? DEFAULT_IMAGE_EXTENSION;

		return `${MCP_ATTACHMENT_NAME_PREFIX}-${Date.now()}-${index}.${extension}`;
	}

	private extractContinuationToolCall(content: string, toolCallId: string): MCPToolCall | null {
		const payload = this.parseContinuationPayload(content);
		if (!this.requiresContinuation(payload)) {
			return null;
		}

		const nextCall =
			this.parseCallEnvelope(payload.next_call_json) ??
			this.parseCallEnvelope(payload.required_tool_arguments_json) ??
			this.parseNextActionCallEnvelope(payload);
		if (!nextCall) {
			return null;
		}

		return {
			id: toolCallId,
			function: {
				name: nextCall.name,
				arguments: JSON.stringify(nextCall.arguments)
			}
		};
	}

	private requiresContinuation(payload: ContinuationPayload): boolean {
		const status = typeof payload.status === 'string' ? payload.status : '';
		const acceptanceStatus =
			typeof payload.acceptance_status === 'string' ? payload.acceptance_status : '';
		if (status === 'needs_continue') {
			return true;
		}

		return (
			acceptanceStatus === 'continue' ||
			payload.task_completion === 'incomplete' ||
			payload.batch_completion === 'incomplete' ||
			payload.content_read_completion === 'incomplete' ||
			payload.goal_status === 'not_complete' ||
			this.parseBooleanish(payload.continue_required) ||
			this.parseBooleanish(payload.auto_continue_required) ||
			payload.assistant_response_allowed === false ||
			payload.final_answer_allowed === false
		);
	}

	private parseContinuationPayload(content: string): ContinuationPayload {
		const trimmed = content.trim();
		if (!trimmed) {
			return {};
		}

		const parsedJson = this.parseJsonLike(trimmed);
		if (parsedJson && typeof parsedJson === 'object' && !Array.isArray(parsedJson)) {
			return parsedJson as ContinuationPayload;
		}

		const payload: Record<string, unknown> = {};
		for (const line of trimmed.split(/\r?\n/)) {
			const normalized = line.trim();
			if (!normalized) {
				continue;
			}

			const separatorIndex = normalized.indexOf('=');
			if (separatorIndex <= 0) {
				continue;
			}

			const key = normalized.slice(0, separatorIndex).trim();
			const rawValue = normalized.slice(separatorIndex + 1).trim();
			if (!key) {
				continue;
			}

			const parsedValue = this.parseJsonLike(rawValue);
			payload[key] = parsedValue ?? rawValue;
		}

		return payload as ContinuationPayload;
	}

	private parseNextActionCallEnvelope(
		payload: ContinuationPayload
	): { name: string; arguments: Record<string, unknown> } | null {
		if (
			typeof payload.next_action_0_tool_name !== 'string' ||
			!payload.next_action_0_tool_name ||
			typeof payload.next_action_0_params_json !== 'string'
		) {
			return null;
		}

		const parsed = this.parseJsonLike(payload.next_action_0_params_json);
		if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
			return null;
		}

		return {
			name: payload.next_action_0_tool_name,
			arguments: parsed as Record<string, unknown>
		};
	}

	private parseCallEnvelope(
		value: unknown
	): { name: string; arguments: Record<string, unknown> } | null {
		const parsed = this.parseJsonLike(value);
		if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
			return null;
		}

		const envelope = parsed as Record<string, unknown>;
		const toolName =
			typeof envelope.tool === 'string'
				? envelope.tool
				: typeof envelope.name === 'string'
					? envelope.name
					: '';
		if (!toolName) {
			return null;
		}

		const args =
			envelope.params && typeof envelope.params === 'object' && !Array.isArray(envelope.params)
				? (envelope.params as Record<string, unknown>)
				: envelope.arguments &&
					  typeof envelope.arguments === 'object' &&
					  !Array.isArray(envelope.arguments)
					? (envelope.arguments as Record<string, unknown>)
					: {};

		return { name: toolName, arguments: args };
	}

	private parseJsonLike(value: unknown): unknown {
		if (typeof value !== 'string') {
			return value;
		}

		const trimmed = value.trim();
		if (!trimmed) {
			return '';
		}

		const lowered = trimmed.toLowerCase();
		if (lowered === 'true') {
			return true;
		}
		if (lowered === 'false') {
			return false;
		}

		try {
			return JSON.parse(trimmed);
		} catch {
			return null;
		}
	}

	private parseBooleanish(value: unknown): boolean {
		if (typeof value === 'boolean') {
			return value;
		}
		if (typeof value === 'string') {
			const lowered = value.trim().toLowerCase();
			return lowered === 'true' || lowered === '1' || lowered === 'yes' || lowered === 'on';
		}
		if (typeof value === 'number') {
			return value !== 0;
		}
		return false;
	}
}

export const agenticStore = new AgenticStore();

export function agenticIsRunning(conversationId: string) {
	return agenticStore.isRunning(conversationId);
}

export function agenticCurrentTurn(conversationId: string) {
	return agenticStore.currentTurn(conversationId);
}

export function agenticTotalToolCalls(conversationId: string) {
	return agenticStore.totalToolCalls(conversationId);
}

export function agenticLastError(conversationId: string) {
	return agenticStore.lastError(conversationId);
}

export function agenticStreamingToolCall(conversationId: string) {
	return agenticStore.streamingToolCall(conversationId);
}

export function agenticIsAnyRunning() {
	return agenticStore.isAnyRunning;
}
