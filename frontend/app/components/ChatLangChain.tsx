import React, { useState } from "react";
import {
  AppendMessage,
  AssistantRuntimeProvider,
  useExternalStoreRuntime,
} from "@assistant-ui/react";
import { useExternalMessageConverter } from "@assistant-ui/react";
import { BaseMessage } from "@langchain/core/messages";
import { useToast } from "../hooks/use-toast";
import { convertLangchainMessages } from "../utils/convert_messages";
import { ThreadChat } from "./chat-interface";
import { SelectModel } from "./SelectModel";
import { ThreadHistory } from "./thread-history";
import { Toaster } from "./ui/toaster";
import { useGraphContext } from "../contexts/GraphContext";

function ChatLangChainComponent(): React.ReactElement {
  const { toast } = useToast();
  const { threadsData, userData, graphData } = useGraphContext();
  const { userId } = userData;
  const { getUserThreads, threadId: currentThread } = threadsData;
  const { messages, handleUserMessage } = graphData;
  const [isRunning, setIsRunning] = useState(false);

  const isSubmitDisabled = !userId || !currentThread;

  async function onNew(message: AppendMessage): Promise<void> {
    if (isSubmitDisabled) {
      let description = "";
      if (!userId) {
        description = "Unable to find user ID. Please try again later.";
      } else if (!currentThread) {
        description =
          "Unable to find current thread ID. Please try again later.";
      }
      toast({
        title: "Failed to send message",
        description,
      });
      return;
    }
    if (message.content[0]?.type !== "text") {
      throw new Error("Only text messages are supported");
    }
    setIsRunning(true);

    try {
      await handleUserMessage(message.content[0].text);
    } finally {
      setIsRunning(false);
      // Re-fetch threads so that the current thread's title is updated.
      await getUserThreads(userId);
    }
  }

  const threadMessages = useExternalMessageConverter<BaseMessage>({
    callback: convertLangchainMessages,
    messages: messages,
    isRunning,
  });

  const runtime = useExternalStoreRuntime({
    messages: threadMessages,
    isRunning,
    onNew,
  });

  return (
    <div className="overflow-hidden w-full flex md:flex-row flex-col relative">
      {messages.length > 0 ? (
        <div className="absolute top-4 right-16 z-40">
          <SelectModel />
        </div>
      ) : null}
      <div>
        <ThreadHistory />
      </div>
      <div className="w-full h-full overflow-hidden">
        <AssistantRuntimeProvider runtime={runtime}>
          <ThreadChat submitDisabled={isSubmitDisabled} messages={messages} />
        </AssistantRuntimeProvider>
      </div>
      <Toaster />
    </div>
  );
}

export const ChatLangChain = React.memo(ChatLangChainComponent);
