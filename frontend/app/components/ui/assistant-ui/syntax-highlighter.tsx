import { memo } from "react";
import { PrismLight as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark, oneLight } from "react-syntax-highlighter/dist/esm/styles/prism";
import { useTheme } from "../../../contexts/ThemeContext";

// Import languages
import python from "react-syntax-highlighter/dist/esm/languages/prism/python";
import typescript from "react-syntax-highlighter/dist/esm/languages/prism/typescript";
import javascript from "react-syntax-highlighter/dist/esm/languages/prism/javascript";
import jsx from "react-syntax-highlighter/dist/esm/languages/prism/jsx";
import tsx from "react-syntax-highlighter/dist/esm/languages/prism/tsx";
import bash from "react-syntax-highlighter/dist/esm/languages/prism/bash";
import json from "react-syntax-highlighter/dist/esm/languages/prism/json";
import csharp from "react-syntax-highlighter/dist/esm/languages/prism/csharp";
import java from "react-syntax-highlighter/dist/esm/languages/prism/java";

// Register languages
SyntaxHighlighter.registerLanguage('python', python);
SyntaxHighlighter.registerLanguage('typescript', typescript);
SyntaxHighlighter.registerLanguage('javascript', javascript);
SyntaxHighlighter.registerLanguage('jsx', jsx);
SyntaxHighlighter.registerLanguage('tsx', tsx);
SyntaxHighlighter.registerLanguage('bash', bash);
SyntaxHighlighter.registerLanguage('json', json);
SyntaxHighlighter.registerLanguage('js', javascript);
SyntaxHighlighter.registerLanguage('ts', typescript);
SyntaxHighlighter.registerLanguage('cs', csharp);
SyntaxHighlighter.registerLanguage('java', java);

interface CodeHighlighterProps {
  children?: string;
  language?: string;
  className?: string;
}

export const CodeHighlighter = memo(function CodeHighlighter({
  children = '',
  language = 'text',
  className
}: CodeHighlighterProps) {
  const { theme } = useTheme();
  const style = theme === 'dark' ? oneDark : oneLight;

  // Clean up the language string and provide a fallback
  const normalizedLanguage = language?.toLowerCase().trim() || 'text';
  
  // For debugging
  // console.log('CodeHighlighter props:', {
  //   language: normalizedLanguage,
  //   children: children?.slice(0, 50), // Show first 50 chars
  //   hasContent: !!children
  // });

  return (
    <div className="rounded-lg overflow-hidden">
      <SyntaxHighlighter
        language={normalizedLanguage}
        style={style}
        customStyle={{
          margin: 0,
          padding: '1rem',
          background: theme === 'dark' ? '#1e1e1e' : '#f5f5f5',
          fontSize: '0.9rem',
        }}
        useInlineStyles={true}
        PreTag="div"
        className={className}
      >
        {children || ''}
      </SyntaxHighlighter>
    </div>
  );
});
