import React from "react";
import { useGraphContext } from "../contexts/GraphContext";
import { ModelOptions } from "../types";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";

const modelOptionsAndLabels: Partial<Record<ModelOptions, string>> = {
  "anthropic/claude-3-5-haiku-20241022": "Claude 3.5 Haiku",
  "openai/gpt-4o-mini": "GPT 4o Mini",
  // "groq/llama3-70b-8192": "Llama3 70b (Groq)",
  "google_genai/gemini-pro": "Gemini Pro",
};

export function SelectModelComponent() {
  const {
    graphData: { selectedModel, setSelectedModel },
  } = useGraphContext();
  return (
    <Select
      onValueChange={(v) => setSelectedModel(v as ModelOptions)}
      value={selectedModel}
      defaultValue="anthropic/claude-3-5-haiku-20241022"
    >
      <SelectTrigger className="w-[180px] border-theme-gray text-theme-gray">
        <SelectValue placeholder="Model" />
      </SelectTrigger>
      <SelectContent className="bg-theme-gray text-theme-gray border-theme-gray">
        {Object.entries(modelOptionsAndLabels).map(([model, label]) => (
          <SelectItem className="hover:bg-theme-gray" key={model} value={model}>
            {label}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

export const SelectModel = React.memo(SelectModelComponent);
