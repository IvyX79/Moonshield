import { DEFAULT_SYSTEM_PROMPT, DEFAULT_TEMPERATURE } from '@/utils/app/const';
import { OpenAIError, OpenAIStream } from '@/utils/server';

import { ChatBody, Message } from '@/types/chat';

// @ts-expect-error
import wasm from '../../node_modules/@dqbd/tiktoken/lite/tiktoken_bg.wasm?module';

import tiktokenModel from '@dqbd/tiktoken/encoders/cl100k_base.json';
import { Tiktoken, init } from '@dqbd/tiktoken/lite/init';

export const config = {
  runtime: 'edge',
};

/**
 * Query the Sci-RAG Engine for scientific document retrieval and citation.
 * Falls back to the legacy document fetch if the rag-engine is unavailable.
 */
async function queryRagEngine(question: string): Promise<{
  answer: string;
  citations: Array<{ title: string; relevance: number; source_type: string }>;
  confidence: number;
}> {
  const ragHost = process.env.RAG_ENGINE_HOST || 'http://rag-engine:8000';

  try {
    const response = await fetch(`${ragHost}/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, top_k: 6 }),
      signal: AbortSignal.timeout(15000),
    });

    if (!response.ok) {
      throw new Error(`RAG engine returned ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.warn('RAG engine unavailable, using legacy document fetch:', error);
    return { answer: '', citations: [], confidence: 0 };
  }
}

const handler = async (req: Request): Promise<Response> => {

  try {
    const { model, messages, key, prompt, temperature } =
      (await req.json()) as ChatBody;

    await init((imports) => WebAssembly.instantiate(wasm, imports));
    const encoding = new Tiktoken(
      tiktokenModel.bpe_ranks,
      tiktokenModel.special_tokens,
      tiktokenModel.pat_str,
    );

    const lastMessage = messages[messages.length - 1];

    // Query the Sci-RAG Engine for document-enhanced answers
    const ragResult = await queryRagEngine(lastMessage.content);

    let promptToSend = prompt;
    if (!promptToSend) {
      promptToSend = DEFAULT_SYSTEM_PROMPT;
    }

    let temperatureToUse = temperature;
    if (temperatureToUse == null) {
      temperatureToUse = DEFAULT_TEMPERATURE;
    }

    const prompt_tokens = encoding.encode(promptToSend);
    let tokenCount = prompt_tokens.length;
    let messagesToSend: Message[] = [];

    encoding.free();

    // If we got a RAG answer with citations, use it directly
    if (ragResult.answer && ragResult.citations.length > 0) {
      // Build a citation appendix
      const citationAppendix = ragResult.citations
        .map((c, i) => `[${i + 1}] ${c.title} (${c.source_type}, confidence: ${(c.relevance * 100).toFixed(0)}%)`)
        .join('\n');

      messagesToSend = [
        {
          role: 'system',
          content: `You are a scientific AI assistant. Use the retrieved information below to answer the user's question. Always cite your sources.

Retrieved Information:
${ragResult.answer}

Citations:
${citationAppendix}

Overall confidence: ${(ragResult.confidence * 100).toFixed(0)}%`,
        },
        {
          role: 'user',
          content: lastMessage.content,
        },
      ];
    } else {
      // Fallback: use direct LLM response without RAG context
      messagesToSend = [
        {
          role: 'user',
          content: lastMessage.content,
        },
      ];
    }

    const stream = await OpenAIStream(
      model,
      promptToSend,
      0,
      key,
      messagesToSend,
    );

    return new Response(stream);
  } catch (error) {
    console.error(error);
    if (error instanceof OpenAIError) {
      return new Response('Error', { status: 500, statusText: error.message });
    } else {
      return new Response('Error', { status: 500 });
    }
  }
};

export default handler;
