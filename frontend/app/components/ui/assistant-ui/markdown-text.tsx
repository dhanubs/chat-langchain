"use client";

import {
  CodeHeaderProps,
  MarkdownTextPrimitive,
  useIsMarkdownCodeBlock,
} from "@assistant-ui/react-markdown";
import remarkGfm from "remark-gfm";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import { FC, memo, useState } from "react";
import { CheckIcon, CopyIcon, ExternalLink } from "lucide-react";

import { TooltipIconButton } from "./tooltip-icon-button";
import { CodeHighlighter } from "./syntax-highlighter";
import { cn } from "../../../utils/cn";

import "katex/dist/katex.min.css";

const MarkdownTextImpl = () => {
  return (
    <MarkdownTextPrimitive
      remarkPlugins={[remarkGfm, remarkMath]}
      rehypePlugins={[rehypeKatex]}
      components={{
        h1: ({ node: _node, className, ...props }) => (
          <h1
            className={cn(
              "mb-8 scroll-m-20 text-4xl font-extrabold tracking-tight last:mb-0 text-theme-gray",
              className,
            )}
            {...props}
          />
        ),
        h2: ({ node: _node, className, ...props }) => (
          <h2
            className={cn(
              "mb-4 mt-8 scroll-m-20 text-3xl font-semibold tracking-tight first:mt-0 last:mb-0 text-theme-gray",
              className,
            )}
            {...props}
          />
        ),
        h3: ({ node: _node, className, ...props }) => (
          <h3
            className={cn(
              "mb-4 mt-6 scroll-m-20 text-2xl font-semibold tracking-tight first:mt-0 last:mb-0 text-theme-gray",
              className,
            )}
            {...props}
          />
        ),
        h4: ({ node: _node, className, ...props }) => (
          <h4
            className={cn(
              "mb-4 mt-6 scroll-m-20 text-xl font-semibold tracking-tight first:mt-0 last:mb-0 text-theme-gray",
              className,
            )}
            {...props}
          />
        ),
        h5: ({ node: _node, className, ...props }) => (
          <h5
            className={cn(
              "my-4 text-lg font-semibold first:mt-0 last:mb-0 text-theme-gray",
              className,
            )}
            {...props}
          />
        ),
        h6: ({ node: _node, className, ...props }) => (
          <h6
            className={cn("my-4 font-semibold first:mt-0 last:mb-0 text-theme-gray", className)}
            {...props}
          />
        ),
        p: ({ node: _node, className, ...props }) => (
          <p
            className={cn(
              "mb-5 mt-5 leading-7 first:mt-0 last:mb-0 text-theme-gray",
              className,
            )}
            {...props}
          />
        ),
        a: ({ node: _node, className, ...props }) => (
          <a
            target="_blank"
            className={cn(
              "text-primary text-blue-400 font-medium underline underline-offset-4 inline-flex items-baseline relative pr-[1.1em]",
              className,
            )}
            {...props}
          >
            {props.children}
            <ExternalLink className="w-3 h-3 absolute top-1 right-1" />
          </a>
        ),
        blockquote: ({ node: _node, className, ...props }) => (
          <blockquote
            className={cn("border-l-2 pl-6 italic text-theme-gray border-theme-gray", className)}
            {...props}
          />
        ),
        ul: ({ node: _node, className, ...props }) => (
          <ul
            className={cn("my-5 ml-6 list-disc [&>li]:mt-2 text-theme-gray", className)}
            {...props}
          />
        ),
        ol: ({ node: _node, className, ...props }) => (
          <ol
            className={cn("my-5 ml-6 list-decimal [&>li]:mt-2 text-theme-gray", className)}
            {...props}
          />
        ),
        hr: ({ node: _node, className, ...props }) => (
          <hr className={cn("my-5 border-theme-gray", className)} {...props} />
        ),
        table: ({ node: _node, className, ...props }) => (
          <table
            className={cn(
              "my-5 w-full border-separate border-spacing-0 overflow-y-auto text-theme-gray",
              className,
            )}
            {...props}
          />
        ),
        th: ({ node: _node, className, ...props }) => (
          <th
            className={cn(
              "bg-theme-gray px-4 py-2 text-left font-bold first:rounded-tl-lg last:rounded-tr-lg [&[align=center]]:text-center [&[align=right]]:text-right",
              className,
            )}
            {...props}
          />
        ),
        td: ({ node: _node, className, ...props }) => (
          <td
            className={cn(
              "border-b border-l border-theme-gray px-4 py-2 text-left last:border-r [&[align=center]]:text-center [&[align=right]]:text-right",
              className,
            )}
            {...props}
          />
        ),
        tr: ({ node: _node, className, ...props }) => (
          <tr
            className={cn(
              "m-0 border-b border-theme-gray p-0 first:border-t [&:last-child>td:first-child]:rounded-bl-lg [&:last-child>td:last-child]:rounded-br-lg",
              className,
            )}
            {...props}
          />
        ),
        sup: ({ node: _node, className, ...props }) => (
          <sup
            className={cn("[&>a]:text-xs [&>a]:no-underline", className)}
            {...props}
          />
        ),
        pre: ({ node: _node, className, ...props }) => {
          return (
            <pre
              className={cn(
                "overflow-x-auto rounded-lg bg-transparent",
                className,
              )}
              {...props}
            />
          );
        },
        code: function Code({ node: _node, className, ...props }) {
          const isCodeBlock = useIsMarkdownCodeBlock();
          // console.log('Code component:', { isCodeBlock, className, props });
          
          const language = className?.split('-')[1] || 'text';

          if (isCodeBlock) {
            return (
              <div className="my-4">
                <CodeHighlighter language={language}>
                  {(props as any).children}
                </CodeHighlighter>
              </div>
            );
          }

          return (
            <code
              className={cn(
                "bg-theme-gray rounded border border-theme-gray font-semibold text-theme-gray px-1.5 py-0.5",
                className
              )}
              {...props}
            />
          );
        },
        CodeHeader,
      }}
    />
  );
};

export const MarkdownText = memo(MarkdownTextImpl);

const CodeHeader: FC<CodeHeaderProps> = ({ language, code }) => {
  const { isCopied, copyToClipboard } = useCopyToClipboard();
  const onCopy = () => {
    if (!code || isCopied) return;
    copyToClipboard(code);
  };

  return (
    <div className="flex items-center justify-between gap-4 rounded-t-lg bg-theme-gray px-4 py-2 text-sm font-semibold text-theme-gray">
      <span className="lowercase [&>span]:text-xs">{language}</span>
      <TooltipIconButton tooltip="Copy" onClick={onCopy}>
        {!isCopied && <CopyIcon className="text-theme-gray" />}
        {isCopied && <CheckIcon className="text-theme-gray" />}
      </TooltipIconButton>
    </div>
  );
};

const useCopyToClipboard = ({
  copiedDuration = 3000,
}: {
  copiedDuration?: number;
} = {}) => {
  const [isCopied, setIsCopied] = useState<boolean>(false);

  const copyToClipboard = (value: string) => {
    if (!value) return;

    navigator.clipboard.writeText(value).then(() => {
      setIsCopied(true);
      setTimeout(() => setIsCopied(false), copiedDuration);
    });
  };

  return { isCopied, copyToClipboard };
};
