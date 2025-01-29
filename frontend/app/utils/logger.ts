import fs from 'fs';
import path from 'path';
import { format } from 'date-fns';

type LogLevel = 'info' | 'error' | 'warn' | 'debug';

interface LogParams {
  [key: string]: any;
}

const logger = {
  info: (message: string, params?: LogParams) => {
    console.log(JSON.stringify({ level: 'info', message, ...params }));
  },
  error: (message: string, params?: LogParams) => {
    console.error(JSON.stringify({ level: 'error', message, ...params }));
  },
  warn: (message: string, params?: LogParams) => {
    console.warn(JSON.stringify({ level: 'warn', message, ...params }));
  },
  debug: (message: string, params?: LogParams) => {
    console.debug(JSON.stringify({ level: 'debug', message, ...params }));
  },
};

export { logger };

class Logger {
  private logDir: string;
  private currentLogFile: string;

  constructor() {
    // Set up logs directory in production
    if (process.env.NODE_ENV === 'production') {
      this.logDir = path.join(process.cwd(), 'logs');
      // Create logs directory if it doesn't exist
      if (!fs.existsSync(this.logDir)) {
        fs.mkdirSync(this.logDir, { recursive: true });
      }
      this.currentLogFile = this.getLogFileName();
    }
  }

  private getLogFileName(): string {
    const date = format(new Date(), 'yyyy-MM-dd');
    return path.join(this.logDir, `api-${date}.log`);
  }

  private formatMessage(level: string, message: string, meta?: any): string {
    const timestamp = new Date().toISOString();
    const formattedMeta = meta ? JSON.stringify(meta) : '';
    return `[${timestamp}] [${level}] ${message} ${formattedMeta}\n`;
  }

  private writeToFile(message: string) {
    // Check if we need to rotate to a new day's log file
    const logFile = this.getLogFileName();
    if (logFile !== this.currentLogFile) {
      this.currentLogFile = logFile;
    }

    fs.appendFileSync(this.currentLogFile, message);
  }

  info(message: string, meta?: any) {
    const formattedMessage = this.formatMessage('INFO', message, meta);
    
    if (process.env.NODE_ENV === 'production') {
      this.writeToFile(formattedMessage);
    } else {
      console.log(formattedMessage);
    }
  }

  error(message: string, error?: any) {
    const meta = error ? {
      error: error.message,
      stack: error.stack,
      ...error
    } : undefined;

    const formattedMessage = this.formatMessage('ERROR', message, meta);
    
    if (process.env.NODE_ENV === 'production') {
      this.writeToFile(formattedMessage);
    } else {
      console.error(formattedMessage);
    }
  }

  warn(message: string, meta?: any) {
    const formattedMessage = this.formatMessage('WARN', message, meta);
    
    if (process.env.NODE_ENV === 'production') {
      this.writeToFile(formattedMessage);
    } else {
      console.warn(formattedMessage);
    }
  }
}

export const loggerInstance = new Logger(); 