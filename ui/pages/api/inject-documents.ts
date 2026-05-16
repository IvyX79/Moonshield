import type { NextApiRequest, NextApiResponse } from 'next';
import { ChromaClient, TransformersEmbeddingFunction } from 'chromadb';
import { IncomingForm } from 'formidable';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import path from 'path';
import { v4 as uuidv4 } from 'uuid';

export const config = {
  api: {
    bodyParser: false,
  },
};

/**
 * Try to upload the document to the Sci-RAG Engine for Llama Index processing.
 * Falls back to legacy Chroma-only ingestion if unavailable.
 */
async function uploadToRagEngine(filePath: string, fileName: string): Promise<boolean> {
  const ragHost = process.env.RAG_ENGINE_HOST || 'http://rag-engine:8000';

  try {
    const fs = await import('fs');
    const buffer = fs.readFileSync(filePath);
    const blob = new Blob([buffer]);
    const formData = new FormData();
    formData.append('file', blob, fileName);

    const response = await fetch(`${ragHost}/documents/upload`, {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(60000),
    });
    return response.ok;
  } catch (error) {
    console.warn('RAG engine unavailable, using legacy Chroma ingestion:', error);
    return false;
  }
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse,
) {
  try {
    if (req.method !== 'POST') {
      return res.status(405).end();
    }

    const form = new IncomingForm();
    form.parse(req, async (err, fields, files) => {
      try {
        if (err) {
          return res.status(400).json({ error: 'Failed to upload file' });
        }

        const file = (files.file || files.pdf);
        const uploadedFile = file instanceof Array ? file[0] : file;
        if (!uploadedFile) {
          return res.status(400).json({ error: 'No file provided' });
        }

        const filePath = uploadedFile.filepath;
        const fileName = uploadedFile.originalFilename || 'document.pdf';

        // Step 1: Try the Sci-RAG Engine (Llama Index + Smart Citations)
        const ragSuccess = await uploadToRagEngine(filePath, fileName);
        if (ragSuccess) {
          return res.status(200).json({
            success: true,
            method: 'rag-engine',
            message: `Document "${fileName}" indexed with Llama Index via rag-engine`,
          });
        }

        // Step 2: Fallback to legacy ChromaDB ingestion
        const client = new ChromaClient({
          path: process.env.CHROMA_PATH || 'http://chroma-server:8000',
        });

        const loader = new PDFLoader(filePath);

        const originalDocs = await loader.load();

        const splitter = new RecursiveCharacterTextSplitter({
          chunkSize: 500,
          chunkOverlap: 100,
        });      

        const docs = await splitter.splitDocuments(originalDocs);
 
        const { ids, metadatas, documentContents } = processDocuments(docs);

        const embedder = new TransformersEmbeddingFunction();
        const collection = await client.getOrCreateCollection({
          name: 'default-collection',
          embeddingFunction: embedder,
        });

        await collection.add({
          ids,
          metadatas,
          documents: documentContents,
        });

        res.status(200).json({
          success: true,
          method: 'chroma-legacy',
          message: 'Documents processed successfully',
          documentCount: ids.length,
        });
      } catch (parseError) {
        console.error(parseError);
        res.status(500).json({ error: 'An error occurred while processing the documents' });
      }
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Server error' });
  }
}

function processDocuments(docs: any) {
  const ids = [];
  const metadatas = [];
  const documentContents = [];

  for (const document of docs) {
    const id = uuidv4();
    ids.push(id);

    const fallbackTitle = path.basename(document.metadata.source);
    const titleFromMetadata = document.metadata.pdf?.info?.Title;

    const title = titleFromMetadata && titleFromMetadata.length > 0 ? titleFromMetadata : fallbackTitle;

    const metadata = {
      title: title,
      page: document.metadata.loc?.pageNumber || 1,
      source: document.metadata.source,
    };
    metadatas.push(metadata);

    documentContents.push(document.pageContent);
  }

  return { ids, metadatas, documentContents };
}
