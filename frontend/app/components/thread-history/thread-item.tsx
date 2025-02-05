import { useState } from "react";
import { Button } from "../ui/button";
import { TooltipIconButton } from "../ui/assistant-ui/tooltip-icon-button";
import { Trash2 } from "lucide-react";

export interface ThreadProps {
  id: string;
  onClick: () => void;
  onDelete: () => void;
  label: string;
  createdAt: Date;
  isSelected?: boolean;
}

export function Thread(props: ThreadProps) {
  const [isHovering, setIsHovering] = useState(false);

  return (
    <div
      className="flex flex-row items-center justify-between w-full px-1 py-0.5 group"
      onMouseEnter={() => setIsHovering(true)}
      onMouseLeave={() => setIsHovering(false)}
    >
      <Button
        className={`hover:bg-theme-gray text-theme-gray-secondary hover:text-theme-gray justify-start items-center flex-grow mr-1 max-w-[calc(100%-40px)] overflow-hidden ${
          props.isSelected ? 'bg-theme-gray text-theme-gray' : ''
        }`}
        size="sm"
        variant="ghost"
        onClick={props.onClick}
      >
        <p className={`truncate text-sm font-light w-full text-left ${
          props.isSelected ? 'text-theme-gray' : 'text-theme-gray-secondary hover:text-theme-gray'
        }`} title={props.label}>
          {props.label}
        </p>
      </Button>
      <div className={`flex-shrink-0 transition-opacity duration-200 ${isHovering ? 'opacity-100' : 'opacity-0'}`}>
        <TooltipIconButton
          tooltip="Delete thread"
          variant="ghost"
          className="hover:bg-red-500/10 dark:hover:bg-red-500/20"
          onClick={props.onDelete}
        >
          <Trash2 className="w-4 h-4 text-red-500 dark:text-red-400" />
        </TooltipIconButton>
      </div>
    </div>
  );
}
