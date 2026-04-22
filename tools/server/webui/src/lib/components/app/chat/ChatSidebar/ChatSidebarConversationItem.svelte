<script lang="ts">
	import {
		Trash2,
		Pencil,
		MoreHorizontal,
		Download,
		Loader2,
		Square,
		GitBranch
	} from '@lucide/svelte';
	import { DropdownMenuActions } from '$lib/components/app';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { FORK_TREE_DEPTH_PADDING } from '$lib/constants';
	import { getAllLoadingChats } from '$lib/stores/chat.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { onMount } from 'svelte';

	interface Props {
		isActive?: boolean;
		depth?: number;
		conversation: DatabaseConversation;
		handleMobileSidebarItemClick?: () => void;
		onDelete?: (id: string) => void;
		onEdit?: (id: string) => void;
		onSelect?: (id: string) => void;
		onStop?: (id: string) => void;
	}

	let {
		conversation,
		handleMobileSidebarItemClick,
		onDelete,
		onEdit,
		onSelect,
		onStop,
		isActive = false,
		depth = 0
	}: Props = $props();

	let renderActionsDropdown = $state(false);
	let dropdownOpen = $state(false);

	let isLoading = $derived(getAllLoadingChats().includes(conversation.id));
	let auditSummary = $derived.by(() => {
		const meta = [];
		if (conversation.currentSourceLabel) meta.push(conversation.currentSourceLabel);
		else if (conversation.currentSourceType) meta.push(conversation.currentSourceType);
		if (conversation.currentTaskState) meta.push(conversation.currentTaskState);
		if (conversation.currentPrimaryIntent && conversation.currentPrimaryIntent !== 'unresolved') {
			meta.push(conversation.currentPrimaryIntent);
		}
		if (conversation.currentTakeoverRelation) meta.push(conversation.currentTakeoverRelation);
		if (conversation.currentTaskGroupId) meta.push(conversation.currentTaskGroupId);
		if (conversation.lastTaskId) meta.push(conversation.lastTaskId);
		if (conversation.sessionId) meta.push(`session ${conversation.sessionId.slice(0, 8)}`);

		const summary = conversation.currentSummary?.trim();
		const resultLabel = summary?.match(/\b(PASS|FAIL|FAILED|BLOCKED|OK)\b/i)?.[1]?.toUpperCase();
		const compactSummary = summary
			? summary.replace(/\s+/g, ' ').replace(/\b(PASS|FAIL|FAILED|BLOCKED|OK)\b[: ]*/gi, '').trim()
			: '';

		if (resultLabel) {
			meta.unshift(resultLabel === 'FAILED' ? 'FAIL' : resultLabel);
		}

		if (summary) {
			const clipped = compactSummary.length > 60 ? `${compactSummary.slice(0, 60)}...` : compactSummary;
			return meta.length > 0 ? `${meta.join(' | ')} | ${clipped}` : clipped;
		}

		return meta.join(' | ');
	});

	function handleEdit(event: Event) {
		event.stopPropagation();
		onEdit?.(conversation.id);
	}

	function handleDelete(event: Event) {
		event.stopPropagation();
		onDelete?.(conversation.id);
	}

	function handleStop(event: Event) {
		event.stopPropagation();
		onStop?.(conversation.id);
	}

	function handleGlobalEditEvent(event: Event) {
		const customEvent = event as CustomEvent<{ conversationId: string }>;

		if (customEvent.detail.conversationId === conversation.id && isActive) {
			handleEdit(event);
		}
	}

	function handleMouseLeave() {
		if (!dropdownOpen) {
			renderActionsDropdown = false;
		}
	}

	function handleMouseOver() {
		renderActionsDropdown = true;
	}

	function handleSelect() {
		onSelect?.(conversation.id);
	}

	$effect(() => {
		if (!dropdownOpen) {
			renderActionsDropdown = false;
		}
	});

	onMount(() => {
		document.addEventListener('edit-active-conversation', handleGlobalEditEvent as EventListener);

		return () => {
			document.removeEventListener(
				'edit-active-conversation',
				handleGlobalEditEvent as EventListener
			);
		};
	});
</script>

<!-- svelte-ignore a11y_mouse_events_have_key_events -->
<button
	class="group flex min-h-9 w-full cursor-pointer items-center justify-between space-x-3 rounded-lg py-1.5 text-left transition-colors hover:bg-foreground/10 {isActive
		? 'bg-foreground/5 text-accent-foreground'
		: ''} px-3"
	onclick={handleSelect}
	onmouseover={handleMouseOver}
	onmouseleave={handleMouseLeave}
>
	<div
		class="flex min-w-0 flex-1 items-center gap-2"
		style:padding-left="{depth * FORK_TREE_DEPTH_PADDING}px"
	>
		{#if depth > 0}
			<Tooltip.Root>
				<Tooltip.Trigger>
					<a
						href="#/chat/{conversation.forkedFromConversationId}"
						class="flex shrink-0 items-center text-muted-foreground transition-colors hover:text-foreground"
					>
						<GitBranch class="h-3.5 w-3.5" />
					</a>
				</Tooltip.Trigger>

				<Tooltip.Content>
					<p>See parent conversation</p>
				</Tooltip.Content>
			</Tooltip.Root>
		{/if}

		{#if isLoading}
			<Tooltip.Root>
				<Tooltip.Trigger>
					<div
						class="stop-button flex h-4 w-4 shrink-0 cursor-pointer items-center justify-center rounded text-muted-foreground transition-colors hover:text-foreground"
						onclick={handleStop}
						onkeydown={(e) => e.key === 'Enter' && handleStop(e)}
						role="button"
						tabindex="0"
						aria-label="Stop generation"
					>
						<Loader2 class="loading-icon h-3.5 w-3.5 animate-spin" />

						<Square class="stop-icon hidden h-3 w-3 fill-current text-destructive" />
					</div>
				</Tooltip.Trigger>

				<Tooltip.Content>
					<p>Stop generation</p>
				</Tooltip.Content>
			</Tooltip.Root>
		{/if}

		<!-- svelte-ignore a11y_click_events_have_key_events -->
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div class="min-w-0 flex-1" onclick={handleMobileSidebarItemClick}>
			<span class="block truncate text-sm font-medium">
				{conversation.name}
			</span>
			{#if auditSummary}
				<span class="block truncate text-[11px] text-muted-foreground/90">
					{auditSummary}
				</span>
			{/if}
		</div>
	</div>

	{#if renderActionsDropdown}
		<div class="actions flex items-center">
			<DropdownMenuActions
				triggerIcon={MoreHorizontal}
				triggerTooltip="More actions"
				bind:open={dropdownOpen}
				actions={[
					{
						icon: Pencil,
						label: 'Edit',
						onclick: handleEdit,
						shortcut: ['shift', 'cmd', 'e']
					},
					{
						icon: Download,
						label: 'Export',
						onclick: (e: Event) => {
							e.stopPropagation();
							conversationsStore.downloadConversation(conversation.id);
						},
						shortcut: ['shift', 'cmd', 's']
					},
					{
						icon: Trash2,
						label: 'Delete',
						onclick: handleDelete,
						variant: 'destructive',
						shortcut: ['shift', 'cmd', 'd'],
						separator: true
					}
				]}
			/>
		</div>
	{/if}
</button>

<style>
	button {
		:global([data-slot='dropdown-menu-trigger']:not([data-state='open'])) {
			opacity: 0;
		}

		&:is(:hover) :global([data-slot='dropdown-menu-trigger']) {
			opacity: 1;
		}
		@media (max-width: 768px) {
			:global([data-slot='dropdown-menu-trigger']) {
				opacity: 1 !important;
			}
		}

		.stop-button {
			:global(.stop-icon) {
				display: none;
			}

			:global(.loading-icon) {
				display: block;
			}
		}

		&:is(:hover) .stop-button {
			:global(.stop-icon) {
				display: block;
			}

			:global(.loading-icon) {
				display: none;
			}
		}
	}
</style>
