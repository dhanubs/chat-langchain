import { DefaultValues, Thread } from "@langchain/langgraph-sdk"

export class ApiClient { 


  private baseUrl: string
  threads: any

  constructor() {
    this.baseUrl = '/api' // This already works with Vite's proxy
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    })

    if (!response.ok) {
      throw new Error(`API call failed: ${response.statusText}`)
    }

    return response.json()
  }

  async searchThreads(params: { metadata: { user_id: string }; limit: number }):
        Promise<Thread<DefaultValues>[] | PromiseLike<Thread<DefaultValues>[]>> {

    return this.request<Thread<DefaultValues>[]>('/threads/search', {
      method: 'POST',
      body: JSON.stringify(params)
    })
  }

  async streamChat(threadId: string, input: string): Promise<ReadableStream<Uint8Array> | null> {
    const response = await fetch(`${this.baseUrl}/chat/${threadId}/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ input }),
    })

    if (!response.ok) {
      throw new Error(`Stream chat failed: ${response.statusText}`)
    }

    return response.body
  }

  async getUserThreads(userId: string) {
    return this.request(`/threads/${userId}`)
  }
  async createThread(userId: string, data: any, threadId?: string) {
    const params = {
      thread_id: threadId,
      metadata: { user_id: userId, ...data }
    };
    return this.request<Thread>('/threads', {
      method: 'POST',
      body: JSON.stringify(params)
    })
  }

  async deleteThread(threadId: string) {
    return this.request(`/threads/${threadId}`, {
      method: 'DELETE'
    })
  }

  async shareRun(runId: string) {
    return this.request(`/runs/share`, {
      method: 'POST',
      body: JSON.stringify({ runId })
    })
  }

  async getThreadById(threadId: string): Promise<Thread> {
    const response = await fetch(`${this.baseUrl}/threads/${threadId}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to fetch thread: ${response.statusText}`);
    }

    return response.json();
  }
}

export const apiClient = new ApiClient() 