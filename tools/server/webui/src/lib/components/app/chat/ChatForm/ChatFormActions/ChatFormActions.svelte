<script lang="ts">
	import { Square } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import {
		ChatFormActionAttachmentsDropdown,
		ChatFormActionAttachmentsSheet,
		ChatFormActionRecord,
		ChatFormActionSubmit,
		McpServersSelector,
		ModelsSelector,
		ModelsSelectorSheet
	} from '$lib/components/app';
	import { SETTINGS_SECTION_TITLES } from '$lib/constants';
	import { mcpStore } from '$lib/stores/mcp.svelte';
	import { getChatSettingsDialogContext } from '$lib/contexts';
	import { FileTypeCategory } from '$lib/enums';
	import { getFileTypeCategory } from '$lib/utils';
	import { config } from '$lib/stores/settings.svelte';
	import { modelsStore, modelOptions, selectedModelId } from '$lib/stores/models.svelte';
	import { isRouterMode, serverError } from '$lib/stores/server.svelte';
	import { chatStore } from '$lib/stores/chat.svelte';
	import { activeMessages, conversationsStore } from '$lib/stores/conversations.svelte';
	import { IsMobile } from '$lib/hooks/is-mobile.svelte';
	import { settingsStore } from '$lib/stores/settings.svelte';

	interface Props {
		canSend?: boolean;
		class?: string;
		disabled?: boolean;
		isLoading?: boolean;
		isRecording?: boolean;
		hasText?: boolean;
		uploadedFiles?: ChatUploadedFile[];
		onFileUpload?: () => void;
		onMicClick?: () => void;
		onStop?: () => void;
		onSystemPromptClick?: () => void;
		onMcpPromptClick?: () => void;
		onMcpResourcesClick?: () => void;
	}

	let {
		canSend = false,
		class: className = '',
		disabled = false,
		isLoading = false,
		isRecording = false,
		hasText = false,
		uploadedFiles = [],
		onFileUpload,
		onMicClick,
		onStop,
		onSystemPromptClick,
		onMcpPromptClick,
		onMcpResourcesClick
	}: Props = $props();

	let currentConfig = $derived(config());
	let isRouter = $derived(isRouterMode());
	let isOffline = $derived(!!serverError());

	let conversationModel = $derived(
		chatStore.getConversationModel(activeMessages() as DatabaseMessage[])
	);

	let lastSyncedConversationModel: string | null = null;

	$effect(() => {
		if (conversationModel && conversationModel !== lastSyncedConversationModel) {
			lastSyncedConversationModel = conversationModel;
			modelsStore.selectModelByName(conversationModel);
		} else if (isRouter && !modelsStore.selectedModelId && modelsStore.loadedModelIds.length > 0) {
			lastSyncedConversationModel = null;
			// auto-select the first loaded model only when nothing is selected yet
			const first = modelOptions().find((m) => modelsStore.loadedModelIds.includes(m.model));
			if (first) modelsStore.selectModelById(first.id);
		}
	});

	let activeModelId = $derived.by(() => {
		const options = modelOptions();

		if (!isRouter) {
			return options.length > 0 ? options[0].model : null;
		}

		const selectedId = selectedModelId();
		if (selectedId) {
			const model = options.find((m) => m.id === selectedId);
			if (model) return model.model;
		}

		if (conversationModel) {
			const model = options.find((m) => m.model === conversationModel);
			if (model) return model.model;
		}

		return null;
	});

	let modelPropsVersion = $state(0); // Used to trigger reactivity after fetch

	$effect(() => {
		if (activeModelId) {
			const cached = modelsStore.getModelProps(activeModelId);

			if (!cached) {
				modelsStore.fetchModelProps(activeModelId).then(() => {
					modelPropsVersion++;
				});
			}
		}
	});

	let hasAudioModality = $derived.by(() => {
		if (activeModelId) {
			void modelPropsVersion;

			return modelsStore.modelSupportsAudio(activeModelId);
		}

		return false;
	});

	let hasVisionModality = $derived.by(() => {
		if (activeModelId) {
			void modelPropsVersion;

			return modelsStore.modelSupportsVision(activeModelId);
		}

		return false;
	});

	let hasAudioAttachments = $derived(
		uploadedFiles.some((file) => getFileTypeCategory(file.type) === FileTypeCategory.AUDIO)
	);
	let shouldShowRecordButton = $derived(
		hasAudioModality && !hasText && !hasAudioAttachments && currentConfig.autoMicOnEmpty
	);

	let hasModelSelected = $derived(!isRouter || !!conversationModel || !!selectedModelId());

	let isSelectedModelInCache = $derived.by(() => {
		if (!isRouter) return true;

		if (conversationModel) {
			return modelOptions().some((option) => option.model === conversationModel);
		}

		const currentModelId = selectedModelId();
		if (!currentModelId) return false;

		return modelOptions().some((option) => option.id === currentModelId);
	});

	let submitTooltip = $derived.by(() => {
		if (!hasModelSelected) {
			return 'Please select a model first';
		}

		if (!isSelectedModelInCache) {
			return 'Selected model is not available, please select another';
		}

		return '';
	});

	let selectorModelRef: ModelsSelector | ModelsSelectorSheet | undefined = $state(undefined);

	let isMobile = new IsMobile();

	export function openModelSelector() {
		selectorModelRef?.open();
	}

	const chatSettingsDialog = getChatSettingsDialogContext();

	let hasMcpPromptsSupport = $derived.by(() => {
		const perChatOverrides = conversationsStore.getAllMcpServerOverrides();

		return mcpStore.hasPromptsCapability(perChatOverrides);
	});

	let hasMcpResourcesSupport = $derived.by(() => {
		const perChatOverrides = conversationsStore.getAllMcpServerOverrides();

		return mcpStore.hasResourcesCapability(perChatOverrides);
	});

	const reasoningStrengthOptions = [
		{ value: 'low', label: 'Low' },
		{ value: 'medium', label: 'Medium' },
		{ value: 'high', label: 'High' }
	] as const;
</script>

<div class="flex w-full items-center gap-3 {className}" style="container-type: inline-size">
	<div class="mr-auto flex items-center gap-2">
		{#if isMobile.current}
			<ChatFormActionAttachmentsSheet
				{disabled}
				{hasAudioModality}
				{hasVisionModality}
				{hasMcpPromptsSupport}
				{hasMcpResourcesSupport}
				{onFileUpload}
				{onSystemPromptClick}
				{onMcpPromptClick}
				{onMcpResourcesClick}
				onMcpSettingsClick={() => chatSettingsDialog.open(SETTINGS_SECTION_TITLES.MCP)}
			/>
		{:else}
			<ChatFormActionAttachmentsDropdown
				{disabled}
				{hasAudioModality}
				{hasVisionModality}
				{hasMcpPromptsSupport}
				{hasMcpResourcesSupport}
				{onFileUpload}
				{onSystemPromptClick}
				{onMcpPromptClick}
				{onMcpResourcesClick}
				onMcpSettingsClick={() => chatSettingsDialog.open(SETTINGS_SECTION_TITLES.MCP)}
			/>
		{/if}

		<McpServersSelector
			{disabled}
			onSettingsClick={() => chatSettingsDialog.open(SETTINGS_SECTION_TITLES.MCP)}
		/>
	</div>

	<div class="ml-auto flex items-center gap-1.5">
		{#if isMobile.current}
			<ModelsSelectorSheet
				disabled={disabled || isOffline}
				bind:this={selectorModelRef}
				currentModel={conversationModel}
				forceForegroundText
				useGlobalSelection
			/>
		{:else}
			<ModelsSelector
				disabled={disabled || isOffline}
				bind:this={selectorModelRef}
				currentModel={conversationModel}
				forceForegroundText
				useGlobalSelection
			/>
		{/if}

		<div class="flex items-center gap-2">
			<select
				aria-label="Reasoning strength"
				class="h-8 min-w-[5.5rem] rounded-full border border-border/60 bg-background px-3 text-xs text-foreground shadow-sm outline-none"
				title="Reasoning strength"
				value={String(currentConfig.reasoningStrengthLevel ?? 'medium')}
				onclick={(e) => e.stopPropagation()}
				onchange={(e) => {
					const value = (e.currentTarget as HTMLSelectElement).value;
					settingsStore.updateConfig('reasoningStrengthLevel', value);
				}}
			>
				{#each reasoningStrengthOptions as level (level.value)}
					<option value={level.value}>{level.label}</option>
				{/each}
			</select>

			<label
				class="flex h-8 items-center gap-2 rounded-full border border-border/60 bg-background px-3 text-[11px] text-foreground shadow-sm"
				title="Skip MCP tool permission confirmation popups by default"
			>
				<Checkbox
					checked={!!currentConfig.mcpAutoAuthorizeTools}
					onCheckedChange={(checked) => {
						const nextValue = !!checked;
						settingsStore.updateConfig('mcpAutoAuthorizeTools', nextValue);
						try {
							localStorage.setItem('codex.mcp.requireApproval', nextValue ? 'false' : 'true');
						} catch {
							// Ignore storage write failures so the visible setting still updates.
						}
					}}
					class="size-3.5"
				/>
				<span class="whitespace-nowrap">Skip MCP Confirm</span>
			</label>
		</div>
	</div>

	{#if isLoading}
		<Button
			type="button"
			variant="secondary"
			onclick={onStop}
			class="group h-8 w-8 rounded-full p-0 hover:bg-destructive/10!"
		>
			<span class="sr-only">Stop</span>

			<Square
				class="h-8 w-8 fill-muted-foreground stroke-muted-foreground group-hover:fill-destructive group-hover:stroke-destructive hover:fill-destructive hover:stroke-destructive"
			/>
		</Button>
	{:else if shouldShowRecordButton}
		<ChatFormActionRecord {disabled} {hasAudioModality} {isLoading} {isRecording} {onMicClick} />
	{:else}
		<ChatFormActionSubmit
			canSend={canSend && hasModelSelected && isSelectedModelInCache}
			{disabled}
			{isLoading}
			tooltipLabel={submitTooltip}
			showErrorState={hasModelSelected && !isSelectedModelInCache}
		/>
	{/if}
</div>
