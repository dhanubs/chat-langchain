import { GraphProvider } from "./contexts/GraphContext";
import { ThemeProvider } from "./contexts/ThemeContext";
import { ThemeToggle } from "./components/ThemeToggle";
import { ChatLangChain } from "./components/ChatLangChain";

function App() {
  return (
    <ThemeProvider>
      <div className="flex flex-col h-full w-full bg-white dark:bg-[#131318] text-gray-900 dark:text-gray-100">
        <ThemeToggle />
        <main className="w-full h-full">
          <GraphProvider>
            <ChatLangChain />
          </GraphProvider>
        </main>
      </div>
    </ThemeProvider>
  );
}

export default App;