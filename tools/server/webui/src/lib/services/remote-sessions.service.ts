import { getJsonHeaders } from '$lib/utils/api-headers';

export class RemoteSessionsService {
	static async list(limit = 50): Promise<ApiRemoteSessionSummary[]> {
		const response = await fetch(`./v1/remote-sessions?limit=${limit}`, {
			headers: getJsonHeaders()
		});
		if (!response.ok) {
			throw new Error(`Failed to list remote sessions (${response.status})`);
		}
		const data = (await response.json()) as ApiRemoteSessionListResponse;
		return data.items || [];
	}

	static async get(sessionId: string): Promise<ApiRemoteSession> {
		const response = await fetch(`./v1/remote-sessions/${encodeURIComponent(sessionId)}`, {
			headers: getJsonHeaders()
		});
		if (!response.ok) {
			throw new Error(`Failed to get remote session ${sessionId} (${response.status})`);
		}
		return (await response.json()) as ApiRemoteSession;
	}

	static async appendTurn(
		sessionId: string,
		body: ApiRemoteSessionTurnRequest
	): Promise<ApiRemoteSessionTurnResponse> {
		const response = await fetch(
			`./v1/remote-sessions/${encodeURIComponent(sessionId)}/append-turn`,
			{
				method: 'POST',
				headers: getJsonHeaders(),
				body: JSON.stringify(body)
			}
		);
		if (!response.ok) {
			const text = await response.text();
			throw new Error(text || `Failed to append remote session turn (${response.status})`);
		}
		return (await response.json()) as ApiRemoteSessionTurnResponse;
	}
}
