import { GraphProvider } from "./contexts/GraphContext";
import { ChatLangChain } from "./components/ChatLangChain";

function App() {
  return (
    <div 
      className="flex flex-col h-full w-full"
      style={{ background: "rgb(38, 38, 41)" }}
    >
      <main className="w-full h-full">
        <GraphProvider>
          <ChatLangChain />
        </GraphProvider>
      </main>
    </div>
  );
}

export default App;