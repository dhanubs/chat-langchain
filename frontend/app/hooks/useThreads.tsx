import { useEffect, useState } from "react";
import { Thread } from "@langchain/langgraph-sdk";
import { getCookie, setCookie } from "../utils/cookies";
import { useToast } from "./use-toast";
import { THREAD_ID_COOKIE_NAME } from "../utils/constants";
import { apiClient } from '../services/api';

// export const runtime = "edge";

export function useThreads(userId: string | undefined) {
  const { toast } = useToast();
  const [isUserThreadsLoading, setIsUserThreadsLoading] = useState(false);
  const [userThreads, setUserThreads] = useState<Thread[]>([]);
  const [threadId, setThreadId] = useState<string>();

  useEffect(() => {
    if (typeof window == "undefined" || !userId) return;
    getUserThreads(userId);
  }, [userId]);

  useEffect(() => {
    if (threadId || typeof window === "undefined" || !userId) return;
    searchOrCreateThread(userId, { title: 'New Thread' });
  }, [userId]);

  const searchOrCreateThread = async (userId: string, data: any) => {
    const threadIdCookie = getCookie(THREAD_ID_COOKIE_NAME);
    if (!threadIdCookie) {
      // No thread ID in cookies, create new thread
      const thread = await createThread(userId, data);
      if (thread?.thread_id) {
        setThreadId(thread.thread_id);
        setCookie(THREAD_ID_COOKIE_NAME, thread.thread_id);
      }
      return thread;
    }

    try {
      // Thread ID exists in cookies, try to get it
      const thread = await getThreadById(threadIdCookie);
      if (thread) {
        // Thread exists, use it regardless of activity
        setThreadId(threadIdCookie);
        return thread;
      }
    } catch (error) {
      // Thread not found or error, create new one
      console.warn('Stored thread not found, creating new:', error);
    }

    // Only create new thread if existing one wasn't found
    const newThread = await createThread(userId, data);
    if (newThread?.thread_id) {
      setThreadId(newThread.thread_id);
      setCookie(THREAD_ID_COOKIE_NAME, newThread.thread_id);
    }
    return newThread;
  };

  const createThread = async (userId: string, data: any) => {
    try {      
      const thread = await apiClient.createThread(userId, data);
      if (!thread || !thread.thread_id) {
        throw new Error("Thread creation failed.");
      }
      return thread;
    } catch (e) {
      console.error("Error creating thread", e);
      toast({
        title: "Error creating thread.",
      });
      throw e;
    }
  };

  const getUserThreads = async (id: string) => {
    setIsUserThreadsLoading(true);
    try {
      const client = apiClient;

      const userThreads = (await client.searchThreads({
        metadata: {
          user_id: id,
        },
        limit: 100,
      })) as Awaited<Thread[]>;

      if (userThreads.length > 0) {
        const lastInArray = userThreads[0];
        const allButLast = userThreads.slice(1, userThreads.length);
        const filteredThreads = allButLast.filter(
          (thread) => thread.values && Object.keys(thread.values).length > 0,
        );
        setUserThreads([...filteredThreads, lastInArray]);
      }
    } finally {
      setIsUserThreadsLoading(false);
    }
  };

  const getThreadById = async (id: string) => {
    const client = apiClient;
    return (await client.getThreadById(id)) as Awaited<Thread>;
  };

  const deleteThread = async (id: string, clearMessages: () => void) => {
    if (!userId) {
      throw new Error("User ID not found");
    }
    setUserThreads((prevThreads) => {
      const newThreads = prevThreads.filter(
        (thread) => thread.thread_id !== id,
      );
      return newThreads;
    });
    if (id === threadId) {
      clearMessages();
      // Create a new thread. Use .then to avoid blocking the UI.
      // Once completed re-fetch threads to update UI.
      searchOrCreateThread(userId, { title: 'New Thread' }).then(async () => {
        await getUserThreads(userId);
      });
    }
    const client = apiClient;
    await client.deleteThread(id);
  };

  return {
    isUserThreadsLoading,
    userThreads,
    setUserThreads,
    getThreadById,
    getUserThreads,
    createThread,
    searchOrCreateThread,
    threadId,
    setThreadId,
    deleteThread,
  };
}
