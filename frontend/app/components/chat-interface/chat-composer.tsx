"use client";

import { ComposerPrimitive, ThreadPrimitive } from "@assistant-ui/react";
import { type FC, useState } from "react";

import { SendHorizontalIcon } from "lucide-react";
import { BaseMessage } from "@langchain/core/messages";
import { TooltipIconButton } from "../ui/assistant-ui/tooltip-icon-button";
import { cn } from "../../utils/cn";

export interface ChatComposerProps {
  messages: BaseMessage[];
  submitDisabled: boolean;
}

const CircleStopIcon = () => {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 16 16"
      fill="currentColor"
      width="16"
      height="16"
    >
      <rect width="10" height="10" x="3" y="3" rx="2" />
    </svg>
  );
};

export const ChatComposer: FC<ChatComposerProps> = (props: ChatComposerProps) => {
  const isEmpty = props.messages.length === 0;
  const [inputValue, setInputValue] = useState("");

  return (
    <ComposerPrimitive.Root
      className={cn(
        "focus-within:border-aui-ring/20 flex w-full items-center md:justify-left justify-center rounded-lg border px-2.5 py-2.5 shadow-sm transition-all duration-300 ease-in-out",
        "bg-theme-gray border-theme-gray",
        isEmpty ? "" : "md:ml-24 ml-3 mb-6",
        isEmpty ? "w-full" : "md:w-[70%] w-[95%] md:max-w-[832px]",
      )}
      onSubmit={() => {
        // Clear the input after submission
        setInputValue("");
      }}
    >
      <ComposerPrimitive.Input
        autoFocus
        placeholder="How can I..."
        rows={1}
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        className="text-theme-gray placeholder:text-theme-gray max-h-40 flex-1 resize-none border-none bg-transparent px-2 py-2 text-sm outline-none focus:ring-0 disabled:cursor-not-allowed"
      />
      <div className="flex-shrink-0">
        <ThreadPrimitive.If running={false} disabled={props.submitDisabled}>
          <ComposerPrimitive.Send asChild>
            <TooltipIconButton
              tooltip="Send message"
              variant="ghost"
              className="w-fit h-fit p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              disabled={props.submitDisabled || !inputValue.trim()}
            >
              <SendHorizontalIcon className="w-5 h-5" />
            </TooltipIconButton>
          </ComposerPrimitive.Send>
        </ThreadPrimitive.If>
        <ThreadPrimitive.If running>
          <ComposerPrimitive.Cancel asChild>
            <TooltipIconButton
              tooltip="Cancel"
              variant="default"
              className="my-1 size-8 p-2 transition-opacity ease-in text-theme-gray"
            >
              <CircleStopIcon />
            </TooltipIconButton>
          </ComposerPrimitive.Cancel>
        </ThreadPrimitive.If>
      </div>
    </ComposerPrimitive.Root>
  );
};
