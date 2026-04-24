import Dexie, { type EntityTable } from 'dexie';
import { findDescendantMessages, uuid, filterByLeafNodeId } from '$lib/utils';
import type { McpServerOverride } from '$lib/types/database';
import type { ApiRemoteSession } from '$lib/types';
import { MessageRole, MessageType } from '$lib/enums';

class LlamacppDatabase extends Dexie {
	conversations!: EntityTable<DatabaseConversation, string>;
	messages!: EntityTable<DatabaseMessage, string>;

	constructor() {
		super('LlamacppWebui');

		this.version(1).stores({
			conversations: 'id, lastModified, currNode, name',
			messages: 'id, convId, type, role, timestamp, parent, children'
		});

		this.version(2).stores({
			conversations: 'id, sessionId, lastModified, currNode, name',
			messages: 'id, convId, sessionId, turnId, taskId, intentKey, type, role, timestamp, parent'
		});
	}
}

const db = new LlamacppDatabase();

export class DatabaseService {
	/**
	 *
	 *
	 * Conversations
	 *
	 *
	 */

	/**
	 * Creates a new conversation.
	 *
	 * @param name - Name of the conversation
	 * @returns The created conversation
	 */
	static async createConversation(name: string): Promise<DatabaseConversation> {
		const conversation: DatabaseConversation = {
			id: uuid(),
			name,
			lastModified: Date.now(),
			currNode: '',
			sessionId: uuid()
		};

		await db.conversations.add(conversation);
		return conversation;
	}

	/**
	 *
	 *
	 * Messages
	 *
	 *
	 */

	/**
	 * Creates a new message branch by adding a message and updating parent/child relationships.
	 * Also updates the conversation's currNode to point to the new message.
	 *
	 * @param message - Message to add (without id)
	 * @param parentId - Parent message ID to attach to
	 * @returns The created message
	 */
	static async createMessageBranch(
		message: Omit<DatabaseMessage, 'id'>,
		parentId: string | null
	): Promise<DatabaseMessage> {
		return await db.transaction('rw', [db.conversations, db.messages], async () => {
			// Handle null parent (root message case)
			if (parentId !== null) {
				const parentMessage = await db.messages.get(parentId);
				if (!parentMessage) {
					throw new Error(`Parent message ${parentId} not found`);
				}
			}

			const newMessage: DatabaseMessage = {
				...message,
				id: uuid(),
				parent: parentId,
				toolCalls: message.toolCalls ?? '',
				children: []
			};

			await db.messages.add(newMessage);

			// Update parent's children array if parent exists
			if (parentId !== null) {
				const parentMessage = await db.messages.get(parentId);
				if (parentMessage) {
					await db.messages.update(parentId, {
						children: [...parentMessage.children, newMessage.id]
					});
				}
			}

			await this.updateConversation(message.convId, {
				currNode: newMessage.id
			});

			return newMessage;
		});
	}

	/**
	 * Creates a root message for a new conversation.
	 * Root messages are not displayed but serve as the tree root for branching.
	 *
	 * @param convId - Conversation ID
	 * @returns The created root message
	 */
	static async createRootMessage(convId: string): Promise<string> {
		const rootMessage: DatabaseMessage = {
			id: uuid(),
			convId,
			type: 'root',
			timestamp: Date.now(),
			role: MessageRole.SYSTEM,
			content: '',
			parent: null,
			toolCalls: '',
			children: []
		};

		await db.messages.add(rootMessage);
		return rootMessage.id;
	}

	/**
	 * Creates a system prompt message for a conversation.
	 *
	 * @param convId - Conversation ID
	 * @param systemPrompt - The system prompt content (must be non-empty)
	 * @param parentId - Parent message ID (typically the root message)
	 * @returns The created system message
	 * @throws Error if systemPrompt is empty
	 */
	static async createSystemMessage(
		convId: string,
		systemPrompt: string,
		parentId: string
	): Promise<DatabaseMessage> {
		const trimmedPrompt = systemPrompt.trim();
		if (!trimmedPrompt) {
			throw new Error('Cannot create system message with empty content');
		}

		const systemMessage: DatabaseMessage = {
			id: uuid(),
			convId,
			type: MessageRole.SYSTEM,
			timestamp: Date.now(),
			role: MessageRole.SYSTEM,
			content: trimmedPrompt,
			parent: parentId,
			children: []
		};

		await db.messages.add(systemMessage);

		const parentMessage = await db.messages.get(parentId);
		if (parentMessage) {
			await db.messages.update(parentId, {
				children: [...parentMessage.children, systemMessage.id]
			});
		}

		return systemMessage;
	}

	/**
	 * Deletes a conversation and all its messages.
	 *
	 * @param id - Conversation ID
	 */
	static async deleteConversation(
		id: string,
		options?: { deleteWithForks?: boolean }
	): Promise<void> {
		await db.transaction('rw', [db.conversations, db.messages], async () => {
			if (options?.deleteWithForks) {
				// Recursively collect all descendant IDs
				const idsToDelete: string[] = [];
				const queue = [id];

				while (queue.length > 0) {
					const parentId = queue.pop()!;
					const children = await db.conversations
						.filter((c) => c.forkedFromConversationId === parentId)
						.toArray();

					for (const child of children) {
						idsToDelete.push(child.id);
						queue.push(child.id);
					}
				}

				for (const forkId of idsToDelete) {
					await db.conversations.delete(forkId);
					await db.messages.where('convId').equals(forkId).delete();
				}
			} else {
				// Reparent direct children to deleted conv's parent
				const conv = await db.conversations.get(id);
				const newParent = conv?.forkedFromConversationId;
				const directChildren = await db.conversations
					.filter((c) => c.forkedFromConversationId === id)
					.toArray();

				for (const child of directChildren) {
					await db.conversations.update(child.id, {
						forkedFromConversationId: newParent ?? undefined
					});
				}
			}

			await db.conversations.delete(id);
			await db.messages.where('convId').equals(id).delete();
		});
	}

	/**
	 * Deletes a message and removes it from its parent's children array.
	 *
	 * @param messageId - ID of the message to delete
	 */
	static async deleteMessage(messageId: string): Promise<void> {
		await db.transaction('rw', db.messages, async () => {
			const message = await db.messages.get(messageId);
			if (!message) return;

			// Remove this message from its parent's children array
			if (message.parent) {
				const parent = await db.messages.get(message.parent);
				if (parent) {
					parent.children = parent.children.filter((childId: string) => childId !== messageId);
					await db.messages.put(parent);
				}
			}

			// Delete the message
			await db.messages.delete(messageId);
		});
	}

	/**
	 * Deletes a message and all its descendant messages (cascading deletion).
	 * This removes the entire branch starting from the specified message.
	 *
	 * @param conversationId - ID of the conversation containing the message
	 * @param messageId - ID of the root message to delete (along with all descendants)
	 * @returns Array of all deleted message IDs
	 */
	static async deleteMessageCascading(
		conversationId: string,
		messageId: string
	): Promise<string[]> {
		return await db.transaction('rw', db.messages, async () => {
			// Get all messages in the conversation to find descendants
			const allMessages = await db.messages.where('convId').equals(conversationId).toArray();

			// Find all descendant messages
			const descendants = findDescendantMessages(allMessages, messageId);
			const allToDelete = [messageId, ...descendants];

			// Get the message to delete for parent cleanup
			const message = await db.messages.get(messageId);
			if (message && message.parent) {
				const parent = await db.messages.get(message.parent);
				if (parent) {
					parent.children = parent.children.filter((childId: string) => childId !== messageId);
					await db.messages.put(parent);
				}
			}

			// Delete all messages in the branch
			await db.messages.bulkDelete(allToDelete);

			return allToDelete;
		});
	}

	/**
	 * Gets all conversations, sorted by last modified time (newest first).
	 *
	 * @returns Array of conversations
	 */
	static async getAllConversations(): Promise<DatabaseConversation[]> {
		return await db.conversations.orderBy('lastModified').reverse().toArray();
	}

	/**
	 * Gets a conversation by ID.
	 *
	 * @param id - Conversation ID
	 * @returns The conversation if found, otherwise undefined
	 */
	static async getConversation(id: string): Promise<DatabaseConversation | undefined> {
		return await db.conversations.get(id);
	}

	static async getConversationBySessionId(
		sessionId: string
	): Promise<DatabaseConversation | undefined> {
		return await db.conversations.where('sessionId').equals(sessionId).first();
	}

	/**
	 * Gets all messages in a conversation, sorted by timestamp (oldest first).
	 *
	 * @param convId - Conversation ID
	 * @returns Array of messages in the conversation
	 */
	static async getConversationMessages(convId: string): Promise<DatabaseMessage[]> {
		return await db.messages.where('convId').equals(convId).sortBy('timestamp');
	}

	static async getMessagesBySessionId(sessionId: string): Promise<DatabaseMessage[]> {
		return await db.messages.where('sessionId').equals(sessionId).sortBy('timestamp');
	}

	static async getMessagesByTaskId(taskId: string): Promise<DatabaseMessage[]> {
		return await db.messages.where('taskId').equals(taskId).sortBy('timestamp');
	}

	static async getMessagesByIntentKey(intentKey: string): Promise<DatabaseMessage[]> {
		return await db.messages.where('intentKey').equals(intentKey).sortBy('timestamp');
	}

	/**
	 * Updates a conversation.
	 *
	 * @param id - Conversation ID
	 * @param updates - Partial updates to apply
	 * @returns Promise that resolves when the conversation is updated
	 */
	static async updateConversation(
		id: string,
		updates: Partial<Omit<DatabaseConversation, 'id'>>
	): Promise<void> {
		await db.conversations.update(id, {
			...updates,
			lastModified: Date.now()
		});
	}

	/**
	 *
	 *
	 * Navigation
	 *
	 *
	 */

	/**
	 * Updates the conversation's current node (active branch).
	 * This determines which conversation path is currently being viewed.
	 *
	 * @param convId - Conversation ID
	 * @param nodeId - Message ID to set as current node
	 */
	static async updateCurrentNode(convId: string, nodeId: string): Promise<void> {
		await this.updateConversation(convId, {
			currNode: nodeId
		});
	}

	/**
	 * Updates a message.
	 *
	 * @param id - Message ID
	 * @param updates - Partial updates to apply
	 * @returns Promise that resolves when the message is updated
	 */
	static async updateMessage(
		id: string,
		updates: Partial<Omit<DatabaseMessage, 'id'>>
	): Promise<void> {
		await db.messages.update(id, updates);
	}

	static async mirrorRemoteSession(session: ApiRemoteSession): Promise<DatabaseConversation> {
		return await db.transaction('rw', [db.conversations, db.messages], async () => {
			const existingConversation = await db.conversations
				.where('sessionId')
				.equals(session.session_id)
				.first();
			const conversationId = existingConversation?.id || `remote:${session.session_id}`;
			const rootId = `remote:${session.session_id}:root`;
			const turns = session.turns || [];
			const lastTurn = turns[turns.length - 1];
			const lastAssistantTurn = [...turns].reverse().find((turn) => turn.assistant_text?.trim());

			await db.messages.where('convId').equals(conversationId).delete();

			const rootMessage: DatabaseMessage = {
				id: rootId,
				convId: conversationId,
				sessionId: session.session_id,
				type: 'root',
				timestamp: session.created_at || session.updated_at || Date.now(),
				role: MessageRole.SYSTEM,
				content: '',
				parent: null,
				toolCalls: '',
				children: []
			};
			await db.messages.put(rootMessage);

			let previousMessageId = rootId;
			const rootChildren: string[] = [];
			let currNode = rootId;

			for (const turn of turns) {
				const userId = `remote:${session.session_id}:user:${turn.turn_id}`;
				const assistantId = `remote:${session.session_id}:assistant:${turn.turn_id}`;
				const taskId = turn.task_id || undefined;
				const evidenceRefs = turn.evidence_ref ? [turn.evidence_ref] : [];

				const userMessage: DatabaseMessage = {
					id: userId,
					convId: conversationId,
					sessionId: session.session_id,
					turnId: turn.turn_id,
					taskId,
					taskGroupId: turn.task_group_id || undefined,
					intentKey: turn.prompt_purpose || undefined,
					evidenceRefs,
					sourceType: turn.source_type || undefined,
					sourceLabel: turn.source_label || undefined,
					sourceDetail: turn.source_detail || undefined,
					handoffFrom: turn.handoff_from || undefined,
					handoffTo: turn.handoff_to || undefined,
					takeoverRelation: turn.takeover_relation || undefined,
					type: MessageType.TEXT,
					timestamp: turn.timestamp || Date.now(),
					role: MessageRole.USER,
					content: turn.user_text || turn.request_payload?.prompt?.toString?.() || '',
					parent: previousMessageId,
					toolCalls: '',
					children: turn.assistant_text?.trim() ? [assistantId] : [],
					dialogSlice: {
						sessionId: session.session_id,
						turnId: turn.turn_id,
						userText: turn.user_text || '',
						assistantText: turn.assistant_text || '',
						summary: turn.summary || undefined,
						expressionKeys: [],
						evidenceRefs,
						createdAt: turn.timestamp || Date.now()
					},
					executionBinding: {
						sessionId: session.session_id,
						turnId: turn.turn_id,
						primaryIntent: turn.prompt_purpose || undefined,
						taskId,
						taskGroupId: turn.task_group_id || undefined,
						sourceType: turn.source_type || undefined,
						sourceLabel: turn.source_label || undefined,
						sourceDetail: turn.source_detail || undefined,
						handoffFrom: turn.handoff_from || undefined,
						handoffTo: turn.handoff_to || undefined,
						takeoverRelation: turn.takeover_relation || undefined,
						evidenceRefs,
						status: turn.confidence,
						toolName: turn.prompt_purpose || undefined
					}
				};
				await db.messages.put(userMessage);

				if (previousMessageId === rootId) {
					rootChildren.push(userId);
				} else {
					await db.messages.update(previousMessageId, { children: [userId] });
				}

				previousMessageId = userId;
				currNode = userId;

				if (turn.assistant_text?.trim()) {
					const assistantMessage: DatabaseMessage = {
						id: assistantId,
						convId: conversationId,
						sessionId: session.session_id,
						turnId: turn.turn_id,
						taskId,
						taskGroupId: turn.task_group_id || undefined,
						intentKey: turn.prompt_purpose || undefined,
						evidenceRefs,
						sourceType: turn.source_type || undefined,
						sourceLabel: turn.source_label || undefined,
						sourceDetail: turn.source_detail || undefined,
						handoffFrom: turn.handoff_from || undefined,
						handoffTo: turn.handoff_to || undefined,
						takeoverRelation: turn.takeover_relation || undefined,
						type: MessageType.TEXT,
						timestamp: (turn.timestamp || Date.now()) + 1,
						role: MessageRole.ASSISTANT,
						content: turn.assistant_text,
						parent: userId,
						toolCalls: '',
						children: [],
						structuredConclusion: {
							task_state: '',
							reasoning_level: turn.reasoning_level || undefined,
							primary_intent: turn.prompt_purpose || undefined,
							evidence_refs: evidenceRefs,
							next_action: turn.next_action || undefined,
							session_id: session.session_id,
							turn_id: turn.turn_id,
							summary: turn.summary || turn.direct_answer || undefined
						},
						dialogSlice: {
							sessionId: session.session_id,
							turnId: turn.turn_id,
							userText: turn.user_text || '',
							assistantText: turn.assistant_text || '',
							summary: turn.summary || undefined,
							expressionKeys: [],
							evidenceRefs,
							createdAt: (turn.timestamp || Date.now()) + 1
						},
						executionBinding: {
							sessionId: session.session_id,
							turnId: turn.turn_id,
							primaryIntent: turn.prompt_purpose || undefined,
							taskId,
							taskGroupId: turn.task_group_id || undefined,
							sourceType: turn.source_type || undefined,
							sourceLabel: turn.source_label || undefined,
							sourceDetail: turn.source_detail || undefined,
							handoffFrom: turn.handoff_from || undefined,
							handoffTo: turn.handoff_to || undefined,
							takeoverRelation: turn.takeover_relation || undefined,
							evidenceRefs,
							status: turn.confidence,
							toolName: turn.prompt_purpose || undefined
						}
					};
					await db.messages.put(assistantMessage);
					currNode = assistantId;
					previousMessageId = assistantId;
				}
			}

			await db.messages.update(rootId, { children: rootChildren });

			const conversation: DatabaseConversation = {
				id: conversationId,
				name: session.title || lastAssistantTurn?.summary || `Remote session ${session.session_id.slice(0, 8)}`,
				lastModified: session.updated_at || Date.now(),
				currNode,
				sessionId: session.session_id,
				lastTurnId: session.last_turn_id || lastTurn?.turn_id,
				currentTaskState: lastTurn ? (lastTurn.write_mode === 'append' ? 'verify' : 'observe') : undefined,
				currentReasoningLevel: lastTurn?.reasoning_level || undefined,
				currentPrimaryIntent: lastTurn?.prompt_purpose || undefined,
				currentSummary: lastTurn?.summary || lastTurn?.direct_answer || undefined,
				lastTaskId: lastTurn?.task_id || undefined,
				currentTaskGroupId: session.task_group_id || lastTurn?.task_group_id || undefined,
				lastEvidenceRefs: lastTurn?.evidence_ref ? [lastTurn.evidence_ref] : [],
				currentExpressionKeys: [],
				currentSourceType: session.source_type || lastTurn?.source_type || undefined,
				currentSourceLabel: lastTurn?.source_label || session.source_type || undefined,
				currentSourceDetail: lastTurn?.source_detail || undefined,
				currentHandoffFrom: session.handoff_from || lastTurn?.handoff_from || undefined,
				currentHandoffTo: session.handoff_to || lastTurn?.handoff_to || undefined,
				currentTakeoverRelation: session.takeover_relation || lastTurn?.takeover_relation || undefined
			};

			await db.conversations.put(conversation);
			return conversation;
		});
	}

	/**
	 *
	 *
	 * Import
	 *
	 *
	 */

	/**
	 * Imports multiple conversations and their messages.
	 * Skips conversations that already exist.
	 *
	 * @param data - Array of { conv, messages } objects
	 */
	static async importConversations(
		data: { conv: DatabaseConversation; messages: DatabaseMessage[] }[]
	): Promise<{ imported: number; skipped: number }> {
		let importedCount = 0;
		let skippedCount = 0;

		return await db.transaction('rw', [db.conversations, db.messages], async () => {
			for (const item of data) {
				const { conv, messages } = item;

				const existing = await db.conversations.get(conv.id);
				if (existing) {
					console.warn(`Conversation "${conv.name}" already exists, skipping...`);
					skippedCount++;
					continue;
				}

				await db.conversations.add(conv);
				for (const msg of messages) {
					await db.messages.put(msg);
				}

				importedCount++;
			}

			return { imported: importedCount, skipped: skippedCount };
		});
	}

	/**
	 *
	 *
	 * Forking
	 *
	 *
	 */

	/**
	 * Forks a conversation at a specific message, creating a new conversation
	 * containing all messages from the root up to (and including) the target message.
	 *
	 * @param sourceConvId - The source conversation ID
	 * @param atMessageId - The message ID to fork at (the new conversation ends here)
	 * @param options - Fork options (name and whether to include attachments)
	 * @returns The newly created conversation
	 */
	static async forkConversation(
		sourceConvId: string,
		atMessageId: string,
		options: { name: string; includeAttachments: boolean }
	): Promise<DatabaseConversation> {
		return await db.transaction('rw', [db.conversations, db.messages], async () => {
			const sourceConv = await db.conversations.get(sourceConvId);
			if (!sourceConv) {
				throw new Error(`Source conversation ${sourceConvId} not found`);
			}

			const allMessages = await db.messages.where('convId').equals(sourceConvId).toArray();

			const pathMessages = filterByLeafNodeId(allMessages, atMessageId, true) as DatabaseMessage[];
			if (pathMessages.length === 0) {
				throw new Error(`Could not resolve message path to ${atMessageId}`);
			}

			const idMap = new Map<string, string>();

			for (const msg of pathMessages) {
				idMap.set(msg.id, uuid());
			}

			const newConvId = uuid();
			const newSessionId = uuid();
			const clonedMessages: DatabaseMessage[] = pathMessages.map((msg) => {
				const newId = idMap.get(msg.id)!;
				const newParent = msg.parent ? (idMap.get(msg.parent) ?? null) : null;
				const newChildren = msg.children
					.filter((childId: string) => idMap.has(childId))
					.map((childId: string) => idMap.get(childId)!);

				return {
					...msg,
					id: newId,
					convId: newConvId,
					sessionId: newSessionId,
					parent: newParent,
					children: newChildren,
					dialogSlice: msg.dialogSlice
						? { ...msg.dialogSlice, sessionId: newSessionId }
						: undefined,
					executionBinding: msg.executionBinding
						? { ...msg.executionBinding, sessionId: newSessionId }
						: undefined,
					extra: options.includeAttachments ? msg.extra : undefined
				};
			});

			const lastClonedMessage = clonedMessages[clonedMessages.length - 1];
			const newConv: DatabaseConversation = {
				id: newConvId,
				name: options.name,
				lastModified: Date.now(),
				currNode: lastClonedMessage.id,
				sessionId: newSessionId,
				forkedFromConversationId: sourceConvId,
				mcpServerOverrides: sourceConv.mcpServerOverrides
					? sourceConv.mcpServerOverrides.map((o: McpServerOverride) => ({
							serverId: o.serverId,
							enabled: o.enabled
						}))
					: undefined
			};

			await db.conversations.add(newConv);

			for (const msg of clonedMessages) {
				await db.messages.add(msg);
			}

			return newConv;
		});
	}
}
