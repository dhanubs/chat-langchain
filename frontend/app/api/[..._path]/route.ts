import { NextRequest, NextResponse } from "next/server";
import { logger } from '@/app/utils/logger';

function getCorsHeaders() {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "*",
  };
}

async function handleRequest(req: NextRequest, method: string) {
  const requestId = crypto.randomUUID();
  const startTime = Date.now();

  logger.info(`Request started`, {
    requestId,
    method,
    url: req.url,
    path: req.nextUrl.pathname,
    userAgent: req.headers.get('user-agent'),
  });

  try {
    const path = req.nextUrl.pathname.replace(/^\/?api\//, "");
    const url = new URL(req.url);
    const searchParams = new URLSearchParams(url.search);
    searchParams.delete("_path");
    searchParams.delete("nxtP_path");
    const queryString = searchParams.toString()
      ? `?${searchParams.toString()}`
      : "";

    const targetUrl = `${process.env.API_BASE_URL}/${path}${queryString}`;
    
    logger.info(`Forwarding request`, {
      requestId,
      targetUrl,
      method,
    });

    const options: RequestInit = {
      method,
      headers: {
        "x-api-key": process.env.LANGCHAIN_API_KEY || "",
        "x-request-id": requestId,
      },
    };

    if (["POST", "PUT", "PATCH"].includes(method)) {
      const body = await req.json();
      options.body = JSON.stringify(body);
      options.headers = {
        ...options.headers,
        "Content-Type": "application/json",
      };
      
      logger.info(`Request body received`, {
        requestId,
        bodyLength: JSON.stringify(body).length,
        bodyPreview: JSON.stringify(body).substring(0, 100) + (JSON.stringify(body).length > 100 ? '...' : ''),
      });
    }

    const res = await fetch(targetUrl, options);

    const duration = Date.now() - startTime;
    logger.info(`Request completed`, {
      requestId,
      status: res.status,
      duration,
    });

    return new NextResponse(res.body, {
      status: res.status,
      statusText: res.statusText,
      headers: {
        ...res.headers,
        ...getCorsHeaders(),
        'x-request-id': requestId,
      },
    });
  } catch (e: any) {
    const duration = Date.now() - startTime;
    logger.error(`Request failed`, {
      requestId,
      duration,
      url: req.url,
      error: e,
    });

    return NextResponse.json(
      { error: e.message },
      { 
        status: e.status ?? 500,
        headers: {
          ...getCorsHeaders(),
          'x-request-id': requestId,
        }
      }
    );
  }
}

export const GET = (req: NextRequest) => handleRequest(req, "GET");
export const POST = (req: NextRequest) => handleRequest(req, "POST");
export const PUT = (req: NextRequest) => handleRequest(req, "PUT");
export const PATCH = (req: NextRequest) => handleRequest(req, "PATCH");
export const DELETE = (req: NextRequest) => handleRequest(req, "DELETE");

// Add a new OPTIONS handler
export const OPTIONS = () => {
  return new NextResponse(null, {
    status: 204,
    headers: {
      ...getCorsHeaders(),
    },
  });
};
